mod render;
mod visuals;

use audio_processor::{log_band_edges, spawn_audio_engine, update_bands, NUM_BANDS};
use render::{FrameData, MeshBuilder, Vertex, Visualizer};
use std::sync::Arc;
use std::time::Instant;
use winit::{
    event::{ElementState, Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowBuilder},
};

/// 4x MSAA. Every visual is built from raw triangles with no per-pixel
/// antialiasing of its own, so without this the thin strokes crawl badly.
const SAMPLE_COUNT: u32 = 4;

struct GpuState {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    vertex_capacity: usize,
    sample_count: u32,
    msaa_view: Option<wgpu::TextureView>,
}

impl GpuState {
    async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let surface = instance
            .create_surface(window)
            .expect("failed to create surface");

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("no suitable GPU adapter found");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .expect("failed to create device");

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);

        // Fall back to no MSAA rather than failing on adapters that don't
        // support 4x for this format.
        let sample_count = if adapter
            .get_texture_format_features(format)
            .flags
            .sample_count_supported(SAMPLE_COUNT)
        {
            SAMPLE_COUNT
        } else {
            1
        };

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("visualizer shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("visualizer pipeline layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("visualizer pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                        wgpu::VertexAttribute {
                            offset: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float32x4,
                        },
                    ],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    // Premultiplied alpha: the shader already scales rgb by a.
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
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
            multisample: wgpu::MultisampleState {
                count: sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        let vertex_capacity = 1 << 16;
        let vertex_buffer = Self::make_vertex_buffer(&device, vertex_capacity);
        let msaa_view = Self::make_msaa_view(&device, &config, sample_count);

        Self {
            surface,
            device,
            queue,
            config,
            pipeline,
            vertex_buffer,
            vertex_capacity,
            sample_count,
            msaa_view,
        }
    }

    fn make_vertex_buffer(device: &wgpu::Device, capacity: usize) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mesh vertex buffer"),
            size: (capacity * std::mem::size_of::<Vertex>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn make_msaa_view(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        sample_count: u32,
    ) -> Option<wgpu::TextureView> {
        if sample_count <= 1 {
            return None;
        }
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("msaa target"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format: config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        Some(texture.create_view(&wgpu::TextureViewDescriptor::default()))
    }

    fn aspect(&self) -> f32 {
        self.config.width as f32 / self.config.height.max(1) as f32
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        self.msaa_view = Self::make_msaa_view(&self.device, &self.config, self.sample_count);
    }

    /// Grow the vertex buffer on demand so a new visual can't silently overflow it.
    fn ensure_capacity(&mut self, needed: usize) {
        if needed <= self.vertex_capacity {
            return;
        }
        let capacity = needed.next_power_of_two();
        self.vertex_buffer = Self::make_vertex_buffer(&self.device, capacity);
        self.vertex_capacity = capacity;
    }

    fn render(&mut self, verts: &[Vertex], clear: wgpu::Color) -> Result<(), wgpu::SurfaceError> {
        self.ensure_capacity(verts.len());

        let frame = self.surface.get_current_texture()?;
        let surface_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        if !verts.is_empty() {
            self.queue
                .write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(verts));
        }

        // With MSAA the pass draws into the multisampled texture and resolves
        // into the swapchain image; without it, straight into the swapchain.
        let (view, resolve_target) = match &self.msaa_view {
            Some(msaa) => (msaa, Some(&surface_view)),
            None => (&surface_view, None),
        };

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("visualizer pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            if !verts.is_empty() {
                pass.set_pipeline(&self.pipeline);
                pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
                pass.draw(0..verts.len() as u32, 0..1);
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        frame.present();
        Ok(())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (metrics, sample_rate, _stream) = spawn_audio_engine()?;
    let edges = log_band_edges(sample_rate);

    let event_loop = EventLoop::new()?;
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("Audio Visualizer")
            .with_inner_size(winit::dpi::LogicalSize::new(1100.0, 620.0))
            .build(&event_loop)?,
    );

    let mut gpu = pollster::block_on(GpuState::new(window.clone()));

    let mut visualizers = visuals::all();
    let mut current = 0usize;

    println!("\nVisualizations (Space or Right/Left to cycle, 1-9 to jump):");
    for (i, v) in visualizers.iter().enumerate() {
        println!("  {}. {}", i + 1, v.name());
    }
    println!();

    let set_title = |window: &Window, v: &dyn Visualizer| {
        window.set_title(&format!("Audio Visualizer — {}", v.name()));
    };
    set_title(&window, visualizers[current].as_ref());

    let mut left_levels = vec![0.0f32; NUM_BANDS];
    let mut right_levels = vec![0.0f32; NUM_BANDS];
    let mut avg_levels = vec![0.0f32; NUM_BANDS];
    let mut mesh = MeshBuilder::new();
    let start = Instant::now();

    event_loop.set_control_flow(ControlFlow::Poll);

    event_loop.run(move |event, elwt| match event {
        Event::WindowEvent { event, window_id } if window_id == window.id() => match event {
            WindowEvent::CloseRequested => elwt.exit(),
            WindowEvent::Resized(size) => gpu.resize(size.width, size.height),
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state != ElementState::Pressed {
                    return;
                }
                let count = visualizers.len();
                let selected = match event.physical_key {
                    PhysicalKey::Code(KeyCode::Space | KeyCode::ArrowRight) => {
                        Some((current + 1) % count)
                    }
                    PhysicalKey::Code(KeyCode::ArrowLeft) => Some((current + count - 1) % count),
                    PhysicalKey::Code(KeyCode::Escape) => {
                        elwt.exit();
                        None
                    }
                    PhysicalKey::Code(code) => {
                        // Digit1..Digit9 are contiguous in the KeyCode enum.
                        let digits = [
                            KeyCode::Digit1,
                            KeyCode::Digit2,
                            KeyCode::Digit3,
                            KeyCode::Digit4,
                            KeyCode::Digit5,
                            KeyCode::Digit6,
                            KeyCode::Digit7,
                            KeyCode::Digit8,
                            KeyCode::Digit9,
                        ];
                        digits
                            .iter()
                            .position(|&d| d == code)
                            .filter(|&i| i < count)
                    }
                    _ => None,
                };

                if let Some(next) = selected {
                    if next != current {
                        current = next;
                        set_title(&window, visualizers[current].as_ref());
                        println!("→ {}", visualizers[current].name());
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                let (left_spec, right_spec, waveform, bass, rms) = {
                    let m = metrics.lock().unwrap();
                    (
                        m.left_spectrum.clone(),
                        m.right_spectrum.clone(),
                        m.waveform.clone(),
                        m.bass_energy,
                        (m.left_rms + m.right_rms) * 0.5,
                    )
                };
                if !left_spec.is_empty() {
                    update_bands(&left_spec, &edges, &mut left_levels);
                    update_bands(&right_spec, &edges, &mut right_levels);
                    for i in 0..NUM_BANDS {
                        avg_levels[i] = (left_levels[i] + right_levels[i]) * 0.5;
                    }
                }

                let background = [
                    0.02 + 0.06 * bass,
                    0.02 + 0.02 * bass,
                    0.05 + 0.10 * bass,
                ];
                let clear = wgpu::Color {
                    r: background[0] as f64,
                    g: background[1] as f64,
                    b: background[2] as f64,
                    a: 1.0,
                };

                let frame = FrameData {
                    bands: &avg_levels,
                    left_bands: &left_levels,
                    right_bands: &right_levels,
                    waveform: &waveform,
                    bass,
                    rms,
                    time: start.elapsed().as_secs_f32(),
                    background,
                };

                mesh.begin(gpu.aspect());
                visualizers[current].draw(&frame, &mut mesh);

                let verts = mesh.vertices().to_vec();
                match gpu.render(&verts, clear) {
                    Ok(()) => {}
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        gpu.resize(gpu.config.width, gpu.config.height)
                    }
                    Err(wgpu::SurfaceError::OutOfMemory) => elwt.exit(),
                    Err(e) => eprintln!("render error: {e:?}"),
                }
            }
            _ => {}
        },
        Event::AboutToWait => window.request_redraw(),
        _ => {}
    })?;

    Ok(())
}
