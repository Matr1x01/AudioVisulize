use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use realfft::RealFftPlanner;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

pub const FFT_SIZE: usize = 1024;
pub const NUM_BINS: usize = FFT_SIZE / 2;

/// Log-spaced display bands the linear FFT bins are grouped into.
pub const NUM_BANDS: usize = 64;
/// Noise floor for dB normalization. Anything quieter maps to zero.
const DB_FLOOR: f32 = -90.0;
/// Hann window halves signal amplitude; scale magnitudes back up.
const HANN_GAIN: f32 = 2.0;

/// Formatted audio metrics frame ready for visualization consumption
#[derive(Debug, Clone, Default)]
pub struct AudioMetrics {
    pub left_spectrum: Vec<f32>,
    pub right_spectrum: Vec<f32>,
    pub left_rms: f32,
    pub right_rms: f32,
    pub bass_energy: f32, // Aggregated low-frequency pulse (20Hz - 250Hz)
    /// Raw mono PCM for the current window, *before* Hann windowing.
    ///
    /// Time-domain visuals (oscilloscopes) need the untapered signal — the
    /// window function fades both ends to zero, which would show up as the
    /// trace collapsing at the edges of the screen.
    pub waveform: Vec<f32>,

    // -- instrumentation ----------------------------------------------------
    // Purely diagnostic; none of this changes the timing being measured.
    /// Monotonic id of the *content*, bumped only when new audio actually
    /// reached the analyser. A pass that saw no new input republishes the
    /// previous `seq`, so the renderer can tell "recomputed identical data"
    /// apart from "genuinely new spectrum" — counting analysis passes instead
    /// would report a healthy update rate that is not really happening.
    pub seq: u64,
    /// When this spectrum was published. `None` before the first pass.
    ///
    /// Age at draw time (`now - produced_at`) is the metric that exposes the
    /// producer/presenter phase drift; comparing rates alone hides it.
    pub produced_at: Option<Instant>,
    /// Input frames the capture callback delivered since the previous analysis
    /// pass — i.e. the FFT hop that actually happened, as opposed to a nominal
    /// configured one. Overlap is `1 - hop_frames / FFT_SIZE`.
    pub hop_frames: u64,
    /// Wall-clock gap between this analysis pass and the previous one.
    pub analysis_dt_ms: f32,
}

/// Lock-free counters written from the real-time capture callback.
///
/// Deliberately atomics rather than fields on `AudioMetrics`: the whole point
/// is to measure how long the callback waits on that mutex, so the measurement
/// cannot itself take it.
#[derive(Debug, Default)]
pub struct AudioProbe {
    /// Input frames delivered by the capture callback since startup.
    pub frames_captured: AtomicU64,
    /// Capture callbacks served.
    pub callbacks: AtomicU64,
    /// Longest single capture-callback execution, microseconds.
    pub callback_max_us: AtomicU64,
    /// Longest time a capture callback spent *waiting for the ring mutex*.
    /// Non-zero means the render/analysis side is blocking the audio thread.
    pub callback_lock_max_us: AtomicU64,
    /// Analysis passes run, including ones that recomputed identical data.
    pub analysis_passes: AtomicU64,
    /// Analysis passes that saw no new input at all — a duplicated window.
    pub stale_analyses: AtomicU64,
    /// Analysis passes that skipped a full window or more of input — audio
    /// that was captured and never analyzed.
    pub dropped_windows: AtomicU64,
}

impl AudioProbe {
    /// Monotonically raise a maximum without taking a lock.
    fn raise_max(cell: &AtomicU64, v: u64) {
        let mut cur = cell.load(Ordering::Relaxed);
        while v > cur {
            match cell.compare_exchange_weak(cur, v, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break,
                Err(observed) => cur = observed,
            }
        }
    }
}

/// Resolve the PulseAudio/PipeWire source to capture system playback from.
///
/// CPAL's ALSA backend cannot enumerate monitor sources — it only sees generic
/// plugin names ("pulse", "default") and raw `hw:` cards, none of which carry
/// system audio. Instead we name the default sink's `.monitor` source and let
/// the ALSA-PulseAudio plugin route to it via `PULSE_SOURCE`.
fn resolve_monitor_source() -> Option<String> {
    // Explicit override wins: e.g. AUDIO_SOURCE=easyeffects_sink.monitor
    if let Ok(name) = std::env::var("AUDIO_SOURCE") {
        if !name.trim().is_empty() {
            return Some(name.trim().to_string());
        }
    }

    let output = Command::new("pactl").arg("get-default-sink").output().ok()?;
    if !output.status.success() {
        return None;
    }

    let sink = String::from_utf8(output.stdout).ok()?.trim().to_string();
    if sink.is_empty() {
        return None;
    }

    Some(format!("{}.monitor", sink))
}

/// Lowest and highest frequency the display covers.
pub fn display_range(sample_rate: u32) -> (f32, f32) {
    let nyquist = sample_rate as f32 / 2.0;
    (20.0, 20_000.0f32.min(nyquist * 0.95))
}

/// Group the linear FFT bins into log-spaced display bands.
///
/// Bins are evenly spaced in Hz, so a linear mapping crams every bass note into
/// the first two columns and spends most of the display on near-silent treble.
/// Log spacing gives each octave equal width, which is how pitch is perceived.
pub fn log_band_edges(sample_rate: u32) -> Vec<(usize, usize)> {
    let bin_hz = sample_rate as f32 / FFT_SIZE as f32;
    let (f_lo, f_hi) = display_range(sample_rate);
    let ratio = (f_hi / f_lo).powf(1.0 / NUM_BANDS as f32);

    let mut edges = Vec::with_capacity(NUM_BANDS);
    let mut f = f_lo;
    for _ in 0..NUM_BANDS {
        let next = f * ratio;
        // Bin 0 is DC offset and never carries musical content.
        let lo = ((f / bin_hz).floor() as usize).clamp(1, NUM_BINS - 1);
        let hi = ((next / bin_hz).ceil() as usize).clamp(lo + 1, NUM_BINS);
        edges.push((lo, hi));
        f = next;
    }
    edges
}

/// Collapse a spectrum into per-band levels, with fast attack and slow decay.
///
/// Without the decay the bars flicker harshly between frames; holding the peak
/// and easing it down is what makes the motion readable.
pub fn update_bands(spectrum: &[f32], edges: &[(usize, usize)], levels: &mut [f32]) {
    for (i, &(lo, hi)) in edges.iter().enumerate() {
        // Peak rather than mean: a single strong tone shouldn't be averaged away
        // by the silent bins sharing its band.
        let peak = spectrum[lo..hi].iter().fold(0.0f32, |a, &b| a.max(b));
        levels[i] = if peak > levels[i] {
            peak
        } else {
            levels[i] * 0.80 + peak * 0.20
        };
    }
}

/// Start capturing system audio and running FFT analysis on a background thread.
///
/// Returns the live metrics handle plus the input stream — the stream must be
/// kept alive by the caller (dropping it stops capture) even though it's never
/// read from directly.
pub type AudioEngine = (
    Arc<Mutex<AudioMetrics>>,
    Arc<AudioProbe>,
    u32,
    cpal::Stream,
);

pub fn spawn_audio_engine() -> Result<AudioEngine, Box<dyn std::error::Error>> {
    let host = cpal::default_host();

    // Point the ALSA pulse plugin at the output monitor before any stream opens.
    // Safe here: still single-threaded, and ALSA reads this when the device opens.
    let monitor = resolve_monitor_source();
    if let Some(source) = &monitor {
        std::env::set_var("PULSE_SOURCE", source);
        println!("Capturing system audio from monitor: {}", source);
    } else {
        eprintln!(
            "WARNING: could not resolve a monitor source (is pactl/PipeWire available?).\n\
             Falling back to the default input device — this is likely a microphone.\n\
             Override explicitly with AUDIO_SOURCE=<source>.monitor"
        );
    }

    // With PULSE_SOURCE set, the "pulse" device carries the monitor stream.
    let device = monitor
        .as_ref()
        .and_then(|_| {
            host.input_devices()
                .ok()?
                .find(|d| matches!(d.name().as_deref(), Ok("pulse")))
        })
        .or_else(|| host.default_input_device())
        .expect("Failed to find a usable audio capture device");

    println!("Capturing audio from device: {}", device.name()?);

    let config = device.default_input_config()?;
    let sample_rate = config.sample_rate().0;
    let channels = config.channels() as usize;

    println!("Sample Rate: {} Hz | Channels: {}", sample_rate, channels);

    // Shared thread-safe state to hold raw PCM samples
    let sample_ring = Arc::new(Mutex::new(Vec::<f32>::with_capacity(FFT_SIZE * channels)));
    let metrics_state = Arc::new(Mutex::new(AudioMetrics::default()));
    let probe = Arc::new(AudioProbe::default());

    let ring_producer = Arc::clone(&sample_ring);
    let probe_producer = Arc::clone(&probe);

    let stream = device.build_input_stream(
        &config.into(),
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            // Instrumentation only. The lock acquisition below is measured, not
            // avoided — establishing how long the real-time thread blocks is
            // the point of this pass.
            let entered = Instant::now();
            let mut lock = ring_producer.lock().unwrap();
            let waited_us = entered.elapsed().as_micros() as u64;

            lock.extend_from_slice(data);

            if lock.len() > FFT_SIZE * channels {
                let overflow = lock.len() - (FFT_SIZE * channels);
                lock.drain(0..overflow);
            }
            drop(lock);

            probe_producer
                .frames_captured
                .fetch_add((data.len() / channels) as u64, Ordering::Relaxed);
            probe_producer.callbacks.fetch_add(1, Ordering::Relaxed);
            AudioProbe::raise_max(&probe_producer.callback_lock_max_us, waited_us);
            AudioProbe::raise_max(
                &probe_producer.callback_max_us,
                entered.elapsed().as_micros() as u64,
            );
        },
        |err| eprintln!("Audio stream error: {}", err),
        None,
    )?;

    stream.play()?;

    let ring_consumer = Arc::clone(&sample_ring);
    let metrics_producer = Arc::clone(&metrics_state);
    let probe_consumer = Arc::clone(&probe);

    thread::spawn(move || {
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(FFT_SIZE);

        let mut left_in = vec![0.0f32; FFT_SIZE];
        let mut right_in = vec![0.0f32; FFT_SIZE];
        let mut left_out = fft.make_output_vec();
        let mut right_out = fft.make_output_vec();

        // Map the 20Hz-250Hz bass band onto bin indices for THIS stream's rate.
        // Bin 0 is DC offset, so the band always starts at 1.
        let bin_hz = sample_rate as f32 / FFT_SIZE as f32;
        let bass_lo = 1usize;
        let bass_hi = (((250.0 / bin_hz).ceil() as usize).max(bass_lo + 1)).min(NUM_BINS);

        let mut seq: u64 = 0;
        let mut last_captured: u64 = 0;
        let mut last_pass = Instant::now();

        loop {
            thread::sleep(Duration::from_millis(16)); // ~60 Hz processing rate

            let pcm_data = {
                let lock = ring_consumer.lock().unwrap();
                if lock.len() < FFT_SIZE * channels {
                    continue; // Wait until full buffer frame fills up
                }
                lock.clone()
            };

            // How far the capture actually advanced since the previous pass.
            // This is the real FFT hop — it is not configured anywhere, it is
            // whatever the wall clock and the callback cadence happened to
            // produce, which is precisely the problem being measured.
            let captured = probe_consumer.frames_captured.load(Ordering::Relaxed);
            let hop_frames = captured.saturating_sub(last_captured);
            last_captured = captured;

            probe_consumer
                .analysis_passes
                .fetch_add(1, Ordering::Relaxed);
            if hop_frames == 0 {
                probe_consumer.stale_analyses.fetch_add(1, Ordering::Relaxed);
            } else {
                // Only new audio makes new content; see `AudioMetrics::seq`.
                seq += 1;
                if hop_frames >= FFT_SIZE as u64 {
                    // The window advanced by at least its own length: there is
                    // no overlap between consecutive analyses, and any excess
                    // beyond FFT_SIZE is audio that is never analysed at all.
                    probe_consumer
                        .dropped_windows
                        .fetch_add(1, Ordering::Relaxed);
                }
            }

            let now = Instant::now();
            let analysis_dt_ms = now.duration_since(last_pass).as_secs_f32() * 1000.0;
            last_pass = now;

            // De-interleave Stereo Channels (L, R, L, R...)
            let mut left_sq_sum = 0.0f32;
            let mut right_sq_sum = 0.0f32;
            let mut waveform = vec![0.0f32; FFT_SIZE];

            for i in 0..FFT_SIZE {
                let l = pcm_data[i * channels];
                let r = if channels > 1 { pcm_data[i * channels + 1] } else { l };

                waveform[i] = (l + r) * 0.5;

                // Apply Hann Windowing to eliminate spectral leakage boundary artifacts
                let hann = 0.5
                    * (1.0
                        - (2.0 * std::f32::consts::PI * i as f32 / (FFT_SIZE - 1) as f32).cos());

                left_in[i] = l * hann;
                right_in[i] = r * hann;

                left_sq_sum += l * l;
                right_sq_sum += r * r;
            }

            fft.process(&mut left_in, &mut left_out).unwrap();
            fft.process(&mut right_in, &mut right_out).unwrap();

            // Calculate Magnitudes and normalize to 0.0-1.0 on a dB scale.
            // dB matches how loudness is perceived and keeps quiet detail visible;
            // a linear magnitude would leave nearly every bar pinned at zero.
            let mut left_spec = vec![0.0f32; NUM_BINS];
            let mut right_spec = vec![0.0f32; NUM_BINS];

            let scale = HANN_GAIN / (FFT_SIZE as f32 / 2.0);
            for i in 0..NUM_BINS {
                let l_mag = left_out[i].norm() * scale;
                let r_mag = right_out[i].norm() * scale;

                let l_db = 20.0 * l_mag.max(1e-10).log10();
                let r_db = 20.0 * r_mag.max(1e-10).log10();

                left_spec[i] = (1.0 - l_db / DB_FLOOR).clamp(0.0, 1.0);
                right_spec[i] = (1.0 - r_db / DB_FLOOR).clamp(0.0, 1.0);
            }

            // Calculate Bass energy over the resolved 20Hz - 250Hz band
            let bass_sum: f32 = left_spec[bass_lo..bass_hi].iter().sum::<f32>()
                + right_spec[bass_lo..bass_hi].iter().sum::<f32>();
            let bass_energy = bass_sum / ((bass_hi - bass_lo) * 2) as f32;

            let mut metrics = metrics_producer.lock().unwrap();
            metrics.left_spectrum = left_spec;
            metrics.right_spectrum = right_spec;
            metrics.left_rms = (left_sq_sum / FFT_SIZE as f32).sqrt();
            metrics.right_rms = (right_sq_sum / FFT_SIZE as f32).sqrt();
            metrics.bass_energy = bass_energy;
            metrics.waveform = waveform;
            metrics.seq = seq;
            metrics.produced_at = Some(Instant::now());
            metrics.hop_frames = hop_frames;
            metrics.analysis_dt_ms = analysis_dt_ms;
        }
    });

    Ok((metrics_state, probe, sample_rate, stream))
}
