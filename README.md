Berlin × Tri-Harmonic — Enhanced Melodic Sequencer + Spheres + 3D Cymatic Pulses + Counterpoint
A real-time, Berlin-style 16-step sequencer with an integrated Tri-Harmonic symbolic core, sphere-driven melodic intelligence, and 3D cymatic pulse visualization. It generates bass/lead + drums, an optional counterpoint voice (S2-derived or exotic scales), and renders cymatic-deformed rings tilted by pitch and coloured by note—all in sync with the audio engine.

Tech stack: Python, PyQt5 UI, VisPy 3D, sounddevice (PortAudio).
Runs locally; no external services.

Highlights
16-step Berlin sequencer: bass/lead + kick/snare/hats, swing, tempo, delay.

Tri-Harmonic core (S1/S2/S3 spheres + torus): symbolic routing with resonance gating, memory decay, rotation.

Melodic Sphere Engine:

Extracts temporal energy, semantic drift, harmonic tension from spheres.

Applies micro-timing, contour pushes, octave strategy, filter modulation, passing tones.

Counterpoint:

Invertible (contrary motion), lagged to lead history.

S2-derived or preset exotic scales (Hirajoshi, Pelog-ish, Messiaen-3, Bhairav-ish).

3D Cymatic Pulses:

Colour by pitch-class, tilt by pitch angle, expanding pooled rings with cymatic deformation.

Efficient pooled rendering; lifetime-controlled, dozens of simultaneous rings.

Race-safe eventing: audio step callback bound after UI initialization; guarded access to flags.

Smoothed fractional delay: time/feedback/mix smoothing to avoid zippering.

Quick Start
bash
Copy
Edit
# 1) Python 3.10–3.12 recommended
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2) The script will auto-install PyQt5, vispy, sounddevice on first run.
python berlin_triharmonic.py
If your system lacks a suitable audio backend, install one:

Windows: WASAPI (default) or ASIO (via ASIO4ALL / manufacturer driver)

macOS: CoreAudio

Linux: ALSA/Pulse/JACK

Controls (Right Panel)
Tempo & Groove

BPM (20–300), Swing (0–50%)

Voices

Bass/Lead Wave: saw, square, sine, triangle, pulse (PWM), supersaw

PWM, Detune (cents), Cutoff (Hz) per voice

Drive (soft saturation), Vibrato (cents / Hz)

Delay

Time (ms), Feedback (%), Mix (%) (all smoothed)

Patterns

Randomize Patterns (bass/lead + drums)

Melodic Sphere Influence

Sphere→Melody global amount

Weights: S1 Rhythm, S2 Melody, S3 Harmony

Harmonic Tension Tracking (on/off)

Counterpoint

Enable

Scale: S2-derived, Hirajoshi, Pelog-ish, Messiaen-3, Bhairav-ish

Lag (steps), Invert (contrary motion), Level

CP Wave + Cutoff

Cymatic Pulses

Enable

Max Rings (pool size)

Ring Lifetime (s)

Stats Panel

Routing metrics, current frequencies, delay settings, counterpoint state

Melodic sphere diagnostics: temporal energy, semantic drift, harmonic tension, tension curve, phase regularity/acceleration, symbol density, phrase memory size

How It Works (Architecture)
TriHarmonicCore

SymbolicSphere (S1 temporal, S2 semantic, S3 harmonic): rotating memory rings with decay and acceptance logic.

ResonanceGate: phase alignment + energy thresholding → sphere routing vs torus.

Metrics: entries/exits, resonance scores, routing efficiency.

BerlinAudio

Real-time DSP via sounddevice.OutputStream (mono).

Oscillators (sine/triangle/square/pulse/saw/supersaw), one-pole LP filters, soft drive.

Step engine: swing, micro-timing, drums (kick/snare/hats), smoothed fractional delay.

Counterpoint: quantized to selectable scales, optional inversion around lead anchor, lagged to lead history.

MelodicSphereEngine

Extracts sphere state → modifies current step:

lead/bass offsets & octave jumps,

micro-timing & dynamics,

filter modulation,

optional passing tones (hook provided).

BerlinTriVis (VisPy + PyQt5)

Turntable 3D: torus wireframe, three sphere wireframes, symbolic particles.

Pooled cymatic rings: colour = pitch class; tilt = pitch; deformation from current bass/lead modes.

UI controls + live stats; event callback dispatch from audio step.

Key Mappings (Internal Logic)
Pitch Colour: HSV hue = (pitch class / 12).

Ring Tilt: angle ∝ semitone distance from root (bounded).

Cymatic Modes: functions of current bass/lead freqs feed sinusoidal deformation of ring radius over θ.

Micro-timing: derived from S1 phase regularity/acceleration (adds subtle push/pull).

Contour Push: S2 semantic drift directs lead offsets; symbol density can enable passing tones.

Tension Strategy: S3 tension brightens/darkens lead filter, controls octave jumps at higher tension.

Preset / Runtime Scales
Presets: Hirajoshi, Pelog-ish, Messiaen-3, Bhairav-ish.

S2-derived: bind a provider to inject a dynamic pitch-class set:

python
Copy
Edit
def get_s2_scale():
    # return list of pitch classes, e.g. [0,2,3,7,8]
    return build_scale_from_classes([0,2,3,7,8])
audio.bind_s2_scale_provider(get_s2_scale)
Troubleshooting
No audio / callback warnings (e.g., “Exception ignored from cffi callback”):

Reduce CPU load: close other apps; lower blocksize in the OutputStream or increase it if underruns occur.

Ensure the selected audio backend supports the current sample rate (44.1 kHz).

Windows: try WASAPI exclusive mode or install an ASIO driver (ASIO4ALL / vendor ASIO).

Stuttering / crackles:

Increase audio block size; reduce delay feedback; reduce visual ring count; lower BPM.

Window opens but blank:

Update GPU drivers; ensure OpenGL 3.3+; try pip install PyOpenGL if needed.

VisPy/Qt errors:

Use app.use_app("pyqt5") (already set). On Linux, ensure PyQt5 and PyOpenGL are present.

Performance Tips
Keep Max Rings moderate (e.g., 16–24).

Use lighter waveforms (sine/triangle) on older CPUs/GPUs.

Lower Vibrato and Detune to reduce per-sample modulation cost.

Avoid extreme feedback or delay time modulation for stability.

Development Notes
Code is organized for extension:

Add scales to PRESET_SCALES or supply runtime S2 providers.

Customize cymatic mapping in _freq_to_modes and _make_pulse.

Implement richer passing tones in MelodicSphereEngine._schedule_passing_tone.

Metrics are exposed in TriHarmonicCore.metrics—feed into logs or OSC/MIDI if desired.

Requirements
The script auto-installs these on first run:

PyQt5

vispy

sounddevice

Recommended Python: 3.10–3.12. Platform: Windows/macOS/Linux.

Run
bash
Copy
Edit
python berlin_triharmonic.py
Safety: start with low master volume and headphones removed while testing.

Roadmap (suggested)
MIDI in/out (clock, note/CC mapping), OSC bridge.

Per-step editing UI; pattern save/load.

Multi-channel audio (stereo, per-voice panning).

Advanced cymatic shaders (GPU) and instanced ring geometry.

Export stems / record to WAV.

License
Choose a license (e.g., MIT) and add it here.

Acknowledgments
Built with PyQt5, VisPy, PortAudio/sounddevice. Inspired by Berlin-school sequencing, psychoacoustics, and cymatics research.
