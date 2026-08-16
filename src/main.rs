mod hud;
mod render;
mod visuals;

use audio_processor::{log_band_edges, spawn_audio_engine, update_bands, FFT_SIZE, NUM_BANDS};
use hud::{FrameSample, Hud, StaticInfo};
use render::{srgb_hex, FrameData, MeshBuilder, Vertex, Visualizer};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
use winit::{
    event::{ElementState, Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowBuilder},
};

/// Default MSAA sample count — off.
///
/// This used to be 4x, and had to be: every visual was raw hard-edged triangles
/// with no antialiasing of its own. That is no longer true. All stroke and fill
/// geometry now carries a one-pixel alpha-feathered fringe which antialiases
/// analytically, and the only unfeathered edges left are the HUD (deliberately
/// pixel-snapped) and `fill_under`'s baseline (always covered by a stroke or
/// butted against an identically-colored fill).
///
/// Measured on Ridge Bed, the heaviest visual, at 1100x620:
///
/// | samples | GPU mean | GPU max  | dropped |
/// |---------|----------|----------|---------|
/// | 4x      | 7.2 ms   | 14.3 ms  | 47      |
/// | 1x      | 3.9 ms   |  9.3 ms  | 0       |
///
/// Side-by-side captures of both the smooth-fill and thin-line cases were
/// indistinguishable, so 4x was costing half the frame budget — and frames —
/// for no visible benefit. Override with `MSAA=1|2|4|8`.
const SAMPLE_COUNT: u32 = 1;

fn requested_sample_count() -> u32 {
    std::env::var("MSAA")
        .ok()
        .and_then(|s| s.trim().parse::<u32>().ok())
        .filter(|n| matches!(n, 1 | 2 | 4 | 8))
        .unwrap_or(SAMPLE_COUNT)
}

/// Wall-clock cost of the render pass, read back from GPU timestamp queries.
///
/// Timestamps are resolved into a mappable buffer and read a frame or two
/// later; blocking on the map would stall the pipeline and corrupt the very
/// number we are trying to measure. One query in flight at a time.
struct GpuTimer {
    query_set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    /// Nanoseconds per timestamp tick, from `Queue::get_timestamp_period`.
    period_ns: f32,
    in_flight: bool,
    /// Set by the map callback once the readback is safe to touch.
    ready: Arc<AtomicBool>,
}

impl GpuTimer {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("render pass timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        });
        // Two u64 timestamps = 16 bytes.
        let resolve = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("timestamp resolve"),
            size: 16,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("timestamp readback"),
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            query_set,
            resolve,
            readback,
            period_ns: queue.get_timestamp_period(),
            in_flight: false,
            ready: Arc::new(AtomicBool::new(false)),
        }
    }

    fn writes(&self) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        Some(wgpu::RenderPassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(0),
            end_of_pass_write_index: Some(1),
        })
    }

    /// Queue the resolve+copy for this frame, if the previous read has landed.
    /// Copying into a mapped buffer is illegal, hence the in-flight guard.
    fn resolve_into(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if self.in_flight {
            return;
        }
        encoder.resolve_query_set(&self.query_set, 0..2, &self.resolve, 0);
        encoder.copy_buffer_to_buffer(&self.resolve, 0, &self.readback, 0, 16);
    }

    /// Start the async map after submit. Never blocks.
    ///
    /// The callback only flips a flag — it may run on a poll thread, and
    /// reading the mapped range from there would race the render thread.
    fn map(&mut self) {
        if self.in_flight {
            return;
        }
        self.in_flight = true;
        self.ready.store(false, Ordering::Release);
        let ready = Arc::clone(&self.ready);
        self.readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |res| {
                ready.store(res.is_ok(), Ordering::Release);
            });
    }

    /// Drain a completed readback, returning the pass duration in ms.
    /// Returns `None` while the map is still outstanding — never blocks.
    fn collect(&mut self, device: &wgpu::Device) -> Option<f32> {
        if !self.in_flight {
            return None;
        }
        device.poll(wgpu::Maintain::Poll);
        if !self.ready.load(Ordering::Acquire) {
            return None;
        }

        let ms = {
            let data = self.readback.slice(..).get_mapped_range();
            let ticks: &[u64] = bytemuck::cast_slice(&data);
            let delta = ticks[1].saturating_sub(ticks[0]);
            delta as f64 * self.period_ns as f64 / 1.0e6
        };
        self.readback.unmap();
        self.in_flight = false;
        self.ready.store(false, Ordering::Release);
        Some(ms as f32)
    }
}

/// Timings collected inside one call to `GpuState::render`.
#[derive(Default)]
struct RenderTiming {
    /// Time blocked in `get_current_texture()`. This is vsync, not work — it
    /// must be excluded from CPU frame time or every frame reads as 16.6 ms.
    wait_ms: f32,
    /// Buffer upload, encoding and submit.
    submit_ms: f32,
    gpu_ms: Option<f32>,
}

struct GpuState {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    vertex_capacity: usize,
    index_buffer: wgpu::Buffer,
    index_capacity: usize,
    sample_count: u32,
    msaa_view: Option<wgpu::TextureView>,
    timer: Option<GpuTimer>,
    present_mode: &'static str,
    available_modes: String,
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

        // TIMESTAMP_QUERY is required to report GPU frame time in the HUD. It
        // is not universally available, so ask for it only when the adapter
        // advertises it and fall back to CPU-only timing otherwise.
        let want_timing = adapter
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY);
        let required_features = if want_timing {
            wgpu::Features::TIMESTAMP_QUERY
        } else {
            wgpu::Features::empty()
        };

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features,
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
        // support the requested count for this format.
        let wanted = requested_sample_count();
        let sample_count = if wanted > 1
            && adapter
                .get_texture_format_features(format)
                .flags
                .sample_count_supported(wanted)
        {
            wanted
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
        let index_capacity = 1 << 18;
        let index_buffer = Self::make_index_buffer(&device, index_capacity);
        let msaa_view = Self::make_msaa_view(&device, &config, sample_count);
        let timer = want_timing.then(|| GpuTimer::new(&device, &queue));

        // Record what the driver actually offers so the HUD can report whether
        // Mailbox was ever an option on this machine.
        let available_modes = caps
            .present_modes
            .iter()
            .map(|m| match m {
                wgpu::PresentMode::Fifo => "FIFO",
                wgpu::PresentMode::FifoRelaxed => "FIFOREL",
                wgpu::PresentMode::Mailbox => "MAILBOX",
                wgpu::PresentMode::Immediate => "IMMEDIATE",
                _ => "AUTO",
            })
            .collect::<Vec<_>>()
            .join(",");

        Self {
            surface,
            device,
            queue,
            config,
            pipeline,
            vertex_buffer,
            vertex_capacity,
            index_buffer,
            index_capacity,
            sample_count,
            msaa_view,
            timer,
            present_mode: "AUTOVSYNC",
            available_modes,
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

    fn make_index_buffer(device: &wgpu::Device, capacity: usize) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mesh index buffer"),
            size: (capacity * std::mem::size_of::<u32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
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

    fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        self.msaa_view = Self::make_msaa_view(&self.device, &self.config, self.sample_count);
    }

    /// Grow the mesh buffers on demand so a new visual can't silently overflow
    /// them. Spline resampling makes vertex counts data-dependent, so this is
    /// load-bearing rather than defensive.
    fn ensure_capacity(&mut self, verts: usize, indices: usize) {
        if verts > self.vertex_capacity {
            let capacity = verts.next_power_of_two();
            self.vertex_buffer = Self::make_vertex_buffer(&self.device, capacity);
            self.vertex_capacity = capacity;
        }
        if indices > self.index_capacity {
            let capacity = indices.next_power_of_two();
            self.index_buffer = Self::make_index_buffer(&self.device, capacity);
            self.index_capacity = capacity;
        }
    }

    fn render(
        &mut self,
        verts: &[Vertex],
        indices: &[u32],
        clear: wgpu::Color,
    ) -> Result<RenderTiming, wgpu::SurfaceError> {
        let mut timing = RenderTiming::default();

        // Collect the previous frame's GPU timestamps before queueing new ones.
        timing.gpu_ms = self
            .timer
            .as_mut()
            .and_then(|t| t.collect(&self.device));

        self.ensure_capacity(verts.len(), indices.len());

        // Everything from here to `get_current_texture` returning is the vsync
        // block on a Fifo swapchain — measured separately so it does not get
        // mistaken for CPU work.
        let wait_start = Instant::now();
        let frame = self.surface.get_current_texture()?;
        timing.wait_ms = wait_start.elapsed().as_secs_f32() * 1000.0;

        let submit_start = Instant::now();
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
        if !indices.is_empty() {
            self.queue
                .write_buffer(&self.index_buffer, 0, bytemuck::cast_slice(indices));
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
                timestamp_writes: self.timer.as_ref().and_then(|t| t.writes()),
                occlusion_query_set: None,
            });

            if !indices.is_empty() {
                pass.set_pipeline(&self.pipeline);
                pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
                pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..indices.len() as u32, 0, 0..1);
            }
        }

        if let Some(t) = self.timer.as_mut() {
            t.resolve_into(&mut encoder);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        timing.submit_ms = submit_start.elapsed().as_secs_f32() * 1000.0;

        frame.present();

        if let Some(t) = self.timer.as_mut() {
            t.map();
        }
        Ok(timing)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (metrics, probe, sample_rate, _stream) = spawn_audio_engine()?;
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
    // `START_VISUAL=<1-based index>` picks the initial visual, so a scripted
    // run can profile a specific one without keyboard input.
    let mut current = std::env::var("START_VISUAL")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .map(|n| n.saturating_sub(1).min(visualizers.len() - 1))
        .unwrap_or(0);

    println!("\nVisualizations (Space or Right/Left to cycle, 1-9 to jump):");
    for (i, v) in visualizers.iter().enumerate() {
        println!("  {}. {}", i + 1, v.name());
    }
    println!("\nF1  toggle debug HUD\nF2  reset HUD statistics\nF3  dump HUD summary to stdout\n");

    let set_title = |window: &Window, v: &dyn Visualizer| {
        window.set_title(&format!("Audio Visualizer — {}", v.name()));
    };
    set_title(&window, visualizers[current].as_ref());

    let mut left_levels = vec![0.0f32; NUM_BANDS];
    let mut right_levels = vec![0.0f32; NUM_BANDS];
    let mut avg_levels = vec![0.0f32; NUM_BANDS];
    let mut mesh = MeshBuilder::new();
    let start = Instant::now();

    let surface = srgb_hex(visuals::SURFACE_HEX);
    let surface_bass = srgb_hex(visuals::SURFACE_BASS_HEX);

    let mut hud = Hud::new();
    let mut band_scratch: Vec<f32> = Vec::with_capacity(NUM_BANDS);
    let mut last_present: Option<Instant> = None;

    // `HUD_AUTODUMP=<seconds>` prints the summary on an interval instead of
    // needing an F3 keypress, so the numbers can be captured from a headless
    // or scripted run. Purely an output trigger; it changes no timing.
    let autodump = std::env::var("HUD_AUTODUMP")
        .ok()
        .and_then(|s| s.trim().parse::<f32>().ok())
        .filter(|s| *s > 0.0)
        .map(std::time::Duration::from_secs_f32);
    let mut last_dump = Instant::now();

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
                    PhysicalKey::Code(KeyCode::F1) => {
                        hud.toggle();
                        None
                    }
                    PhysicalKey::Code(KeyCode::F2) => {
                        hud.reset();
                        probe.callback_max_us.store(0, Ordering::Relaxed);
                        probe.callback_lock_max_us.store(0, Ordering::Relaxed);
                        println!("HUD statistics reset");
                        None
                    }
                    PhysicalKey::Code(KeyCode::F3) => {
                        hud.dump(&probe);
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
                let frame_start = Instant::now();

                // Measure the render thread's wait on the mutex the real-time
                // audio callback also takes — this is the contention half of
                // the priority-inversion question.
                let lock_start = Instant::now();
                let guard = metrics.lock().unwrap();
                let lock_wait_ms = lock_start.elapsed().as_secs_f32() * 1000.0;

                let (left_spec, right_spec, waveform, bass, rms, seq, produced_at, hop_frames) = {
                    let m = guard;
                    (
                        m.left_spectrum.clone(),
                        m.right_spectrum.clone(),
                        m.waveform.clone(),
                        m.bass_energy,
                        (m.left_rms + m.right_rms) * 0.5,
                        m.seq,
                        m.produced_at,
                        m.hop_frames,
                    )
                };

                // Age of the data this frame is about to draw. A pipeline that
                // interpolates to presentation time holds this flat; a
                // free-running producer makes it ramp and wrap.
                let spectrum_age_ms = produced_at
                    .map(|t| frame_start.saturating_duration_since(t).as_secs_f32() * 1000.0)
                    .unwrap_or(0.0);

                if !left_spec.is_empty() {
                    update_bands(&left_spec, &edges, &mut left_levels);
                    update_bands(&right_spec, &edges, &mut right_levels);
                    for i in 0..NUM_BANDS {
                        avg_levels[i] = (left_levels[i] + right_levels[i]) * 0.5;
                    }
                }

                // Material dark surface, lifted very slightly toward Deep
                // Purple 900 by bass. Mixed in linear space, which is what the
                // sRGB surface expects the shader to write.
                //
                // The mix factor is small on purpose: #311B92 is a saturated
                // hue and the eye is being asked to read low-alpha glow against
                // this, so anything above ~0.15 stops reading as "dark surface
                // with a bass lift" and starts reading as a purple background.
                // `bass_energy` is a normalized dB figure that already sits
                // high for ordinary material, which amplifies any factor here.
                let background = render::mix(surface, surface_bass, bass.clamp(0.0, 1.0) * 0.12);
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

                mesh.begin(gpu.config.width as f32, gpu.config.height as f32);
                visualizers[current].draw(&frame, &mut mesh);

                // Scene geometry only — the HUD's own triangles are counted
                // separately so the overlay does not inflate the number it is
                // reporting.
                let scene_verts = mesh.vertices().len();
                let scene_indices = mesh.indices().len();

                let info = StaticInfo {
                    sample_count: gpu.sample_count,
                    fft_size: FFT_SIZE,
                    num_bands: NUM_BANDS,
                    present_mode: gpu.present_mode,
                    available_modes: gpu.available_modes.clone(),
                    gpu_timing: gpu.timer.is_some(),
                    subdivisions: 10,
                    feathered: true,
                    visualizer: visualizers[current].name(),
                };
                hud.note_bands(&avg_levels, &mut band_scratch);
                hud.note_scene_geometry(scene_verts, scene_indices);
                hud.draw(&mut mesh, gpu.config.width, gpu.config.height, &info);

                let cpu_build_ms = frame_start.elapsed().as_secs_f32() * 1000.0;
                // No staging copy: `mesh` and `gpu` are separate locals, so the
                // mesh buffers can be borrowed straight through to the upload.
                let cpu_copy_ms = 0.0;

                let mut sample = FrameSample {
                    cpu_build_ms,
                    cpu_copy_ms,
                    lock_wait_ms,
                    spectrum_age_ms,
                    spectrum_seq: seq,
                    hop_frames,
                    scene_verts,
                    scene_indices,
                    visualizer: visualizers[current].name(),
                    ..Default::default()
                };

                match gpu.render(mesh.vertices(), mesh.indices(), clear) {
                    Ok(timing) => {
                        sample.wait_ms = timing.wait_ms;
                        sample.cpu_submit_ms = timing.submit_ms;
                        if let Some(ms) = timing.gpu_ms {
                            hud.push_gpu_ms(ms);
                        }
                        let now = Instant::now();
                        sample.present_dt_ms = last_present
                            .map(|p| now.duration_since(p).as_secs_f32() * 1000.0)
                            .unwrap_or(0.0);
                        last_present = Some(now);
                    }
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        gpu.resize(gpu.config.width, gpu.config.height)
                    }
                    Err(wgpu::SurfaceError::OutOfMemory) => elwt.exit(),
                    Err(e) => eprintln!("render error: {e:?}"),
                }

                hud.record(&sample);

                if let Some(every) = autodump {
                    if last_dump.elapsed() >= every {
                        hud.dump(&probe);
                        last_dump = Instant::now();
                    }
                }
            }
            _ => {}
        },
        Event::AboutToWait => window.request_redraw(),
        _ => {}
    })?;

    Ok(())
}
