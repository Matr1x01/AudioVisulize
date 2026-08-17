//! Bloom post-process: bright-pass threshold, separable blur, additive
//! composite.
//!
//! The scene pass ([`crate::render`]) fakes glow per-stroke by stacking
//! additively-blended wide strokes under a core line — cheap and good enough
//! for a single curve, but it can't bleed light across separate shapes (a
//! bright bar and its neighbor never brighten each other) the way a real
//! post-process bloom does. Measured headroom justifies the extra passes: the
//! GPU sits at a few ms of a 16.6ms budget with nothing else contending for
//! it (see the MSAA note in `main.rs`).
//!
//! Pipeline: scene renders into an offscreen texture instead of the
//! swapchain -> threshold+downsample to half resolution -> separable
//! Gaussian blur (horizontal, then vertical) -> composite (scene + blurred
//! bright-pass) into the actual swapchain image.

use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Params {
    threshold: f32,
    intensity: f32,
    dir: [f32; 2],
}

/// Luminance a pixel must exceed before it contributes to the glow. Roughly
/// where the shared ramp's teal/ice stops sit — the deep-violet base of every
/// visual should not bloom, only its bright crest.
const THRESHOLD: f32 = 0.55;
/// How strongly the blurred bright-pass is added back over the scene.
const INTENSITY: f32 = 0.85;

fn requested_intensity() -> f32 {
    std::env::var("BLOOM_INTENSITY")
        .ok()
        .and_then(|s| s.trim().parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
        .unwrap_or(INTENSITY)
}

pub struct Bloom {
    format: wgpu::TextureFormat,
    scene_view: wgpu::TextureView,
    bright_view: wgpu::TextureView,
    blur_a_view: wgpu::TextureView,
    blur_b_view: wgpu::TextureView,
    half_w: u32,
    half_h: u32,

    threshold_pipeline: wgpu::RenderPipeline,
    blur_pipeline: wgpu::RenderPipeline,
    composite_pipeline: wgpu::RenderPipeline,

    threshold_buf: wgpu::Buffer,
    blur_h_buf: wgpu::Buffer,
    blur_v_buf: wgpu::Buffer,
    composite_buf: wgpu::Buffer,

    threshold_bg: wgpu::BindGroup,
    blur_h_bg: wgpu::BindGroup,
    blur_v_bg: wgpu::BindGroup,
    composite_bg: wgpu::BindGroup,

    intensity: f32,
}

impl Bloom {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat, width: u32, height: u32) -> Self {
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("bloom sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let sampling_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bloom sampling bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let composite_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bloom composite bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let sample_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bloom sample shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("bloom_sample.wgsl").into()),
        });
        let composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bloom composite shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("bloom_composite.wgsl").into()),
        });

        let sampling_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bloom sampling pipeline layout"),
            bind_group_layouts: &[&sampling_bgl],
            push_constant_ranges: &[],
        });
        let composite_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bloom composite pipeline layout"),
            bind_group_layouts: &[&composite_bgl],
            push_constant_ranges: &[],
        });

        let fullscreen_pipeline = |label: &str,
                                    layout: &wgpu::PipelineLayout,
                                    module: &wgpu::ShaderModule,
                                    entry_point: &'static str,
                                    target_format: wgpu::TextureFormat| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module,
                    entry_point: "vs_fullscreen",
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module,
                    entry_point,
                    targets: &[Some(wgpu::ColorTargetState {
                        format: target_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            })
        };

        let threshold_pipeline = fullscreen_pipeline(
            "bloom threshold pipeline",
            &sampling_layout,
            &sample_shader,
            "fs_threshold",
            format,
        );
        let blur_pipeline = fullscreen_pipeline(
            "bloom blur pipeline",
            &sampling_layout,
            &sample_shader,
            "fs_blur",
            format,
        );
        let composite_pipeline = fullscreen_pipeline(
            "bloom composite pipeline",
            &composite_layout,
            &composite_shader,
            "fs_composite",
            format,
        );

        let make_uniform = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: std::mem::size_of::<Params>() as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let threshold_buf = make_uniform("bloom threshold params");
        let blur_h_buf = make_uniform("bloom blur h params");
        let blur_v_buf = make_uniform("bloom blur v params");
        let composite_buf = make_uniform("bloom composite params");

        let scene_view = Self::make_target(device, format, width, height, "bloom scene");
        let (half_w, half_h) = Self::half_size(width, height);
        let bright_view = Self::make_target(device, format, half_w, half_h, "bloom bright");
        let blur_a_view = Self::make_target(device, format, half_w, half_h, "bloom blur a");
        let blur_b_view = Self::make_target(device, format, half_w, half_h, "bloom blur b");

        let sampling_bind_group = |label: &str, view: &wgpu::TextureView, buf: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &sampling_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buf.as_entire_binding(),
                    },
                ],
            })
        };
        let threshold_bg = sampling_bind_group("bloom threshold bg", &scene_view, &threshold_buf);
        let blur_h_bg = sampling_bind_group("bloom blur h bg", &bright_view, &blur_h_buf);
        let blur_v_bg = sampling_bind_group("bloom blur v bg", &blur_a_view, &blur_v_buf);
        let composite_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bloom composite bg"),
            layout: &composite_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&scene_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&blur_b_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: composite_buf.as_entire_binding(),
                },
            ],
        });

        Self {
            format,
            scene_view,
            bright_view,
            blur_a_view,
            blur_b_view,
            half_w,
            half_h,
            threshold_pipeline,
            blur_pipeline,
            composite_pipeline,
            threshold_buf,
            blur_h_buf,
            blur_v_buf,
            composite_buf,
            threshold_bg,
            blur_h_bg,
            blur_v_bg,
            composite_bg,
            intensity: requested_intensity(),
        }
    }

    fn half_size(width: u32, height: u32) -> (u32, u32) {
        ((width / 2).max(1), (height / 2).max(1))
    }

    /// Builds the texture and immediately discards the handle, keeping only
    /// the view — the same pattern `main.rs` uses for the MSAA target. A
    /// `TextureView` holds its own reference to the underlying resource, so
    /// the parent `Texture` need not be kept alive on the Rust side.
    fn make_target(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        label: &str,
    ) -> wgpu::TextureView {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        tex.create_view(&wgpu::TextureViewDescriptor::default())
    }

    /// Recreate every offscreen target and bind group at the new resolution.
    /// Simplest correct approach on resize, which is rare relative to frames;
    /// rebuilding `Bloom::new`'s output wholesale avoids maintaining two
    /// separate "build" and "rebuild" code paths that could drift apart.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        *self = Self::new(device, self.format, width, height);
    }

    /// Run the threshold, blur and composite passes. `target` is the actual
    /// swapchain view; the scene itself must already have been rendered into
    /// [`Bloom::scene_view`] earlier in this same encoder.
    pub fn render(&self, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, target: &wgpu::TextureView) {
        let (half_w, half_h) = (self.half_w, self.half_h);

        queue.write_buffer(
            &self.threshold_buf,
            0,
            bytemuck::bytes_of(&Params {
                threshold: THRESHOLD,
                intensity: 0.0,
                dir: [0.0, 0.0],
            }),
        );
        queue.write_buffer(
            &self.blur_h_buf,
            0,
            bytemuck::bytes_of(&Params {
                threshold: 0.0,
                intensity: 0.0,
                dir: [1.0 / half_w as f32, 0.0],
            }),
        );
        queue.write_buffer(
            &self.blur_v_buf,
            0,
            bytemuck::bytes_of(&Params {
                threshold: 0.0,
                intensity: 0.0,
                dir: [0.0, 1.0 / half_h as f32],
            }),
        );
        queue.write_buffer(
            &self.composite_buf,
            0,
            bytemuck::bytes_of(&Params {
                threshold: 0.0,
                intensity: self.intensity,
                dir: [0.0, 0.0],
            }),
        );

        let fullscreen_pass = |encoder: &mut wgpu::CommandEncoder,
                                label: &str,
                                view: &wgpu::TextureView,
                                pipeline: &wgpu::RenderPipeline,
                                bind_group: &wgpu::BindGroup| {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(label),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.draw(0..3, 0..1);
        };

        fullscreen_pass(
            encoder,
            "bloom threshold",
            &self.bright_view,
            &self.threshold_pipeline,
            &self.threshold_bg,
        );
        fullscreen_pass(
            encoder,
            "bloom blur h",
            &self.blur_a_view,
            &self.blur_pipeline,
            &self.blur_h_bg,
        );
        fullscreen_pass(
            encoder,
            "bloom blur v",
            &self.blur_b_view,
            &self.blur_pipeline,
            &self.blur_v_bg,
        );
        fullscreen_pass(
            encoder,
            "bloom composite",
            target,
            &self.composite_pipeline,
            &self.composite_bg,
        );
    }

    pub fn scene_view(&self) -> &wgpu::TextureView {
        &self.scene_view
    }
}
