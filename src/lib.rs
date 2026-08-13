use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use realfft::RealFftPlanner;
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

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
pub fn spawn_audio_engine(
) -> Result<(Arc<Mutex<AudioMetrics>>, u32, cpal::Stream), Box<dyn std::error::Error>> {
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

    let ring_producer = Arc::clone(&sample_ring);

    let stream = device.build_input_stream(
        &config.into(),
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            let mut lock = ring_producer.lock().unwrap();
            lock.extend_from_slice(data);

            if lock.len() > FFT_SIZE * channels {
                let overflow = lock.len() - (FFT_SIZE * channels);
                lock.drain(0..overflow);
            }
        },
        |err| eprintln!("Audio stream error: {}", err),
        None,
    )?;

    stream.play()?;

    let ring_consumer = Arc::clone(&sample_ring);
    let metrics_producer = Arc::clone(&metrics_state);

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

        loop {
            thread::sleep(Duration::from_millis(16)); // ~60 Hz processing rate

            let pcm_data = {
                let lock = ring_consumer.lock().unwrap();
                if lock.len() < FFT_SIZE * channels {
                    continue; // Wait until full buffer frame fills up
                }
                lock.clone()
            };

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
        }
    });

    Ok((metrics_state, sample_rate, stream))
}
