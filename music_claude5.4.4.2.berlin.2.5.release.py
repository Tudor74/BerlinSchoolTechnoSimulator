"""
Berlin × Tri-Harmonic — Enhanced Melodic Sequencer + Spheres + 3D Cymatic Pulses + Counterpoint
----------------------------------------------------------------------------------------------------
- 16-step Berlin-style sequencer (bass/lead + kick/snare/hats)
- Full Tri-Harmonic routing/visuals (S1/S2/S3 spheres, torus, *3D cymatic pulse rings*)
- Enhanced melodic generation driven by sphere data and musical intelligence
- Counterpoint voice: S2-derived or exotic scales; full controls
- Smoothed fractional delay (wet/dry/feedback/time)
- Pulse rings: pooled, cymatic-deformed, **tilted in 3D by pitch**, and **coloured by note**
- Sphere-driven melodic contour, tension tracking, micro-timing
- Randomize Patterns control
- Race-safe: event callback hooked after UI/flags; guarded getattr in _on_step

Requires: PyQt5, vispy, sounddevice (auto-installs)
"""

import sys, time, logging, threading
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, List

import numpy as np

# -----------------------------------------------------------------------------#
# Dependency bootstrap
# -----------------------------------------------------------------------------#
def _ensure_pkg(mod_name: str, pip_name: Optional[str] = None):
    try:
        __import__(mod_name)
    except ImportError:
        print(f"[setup] {mod_name} not found. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name or mod_name])

_ensure_pkg("PyQt5", "PyQt5")
_ensure_pkg("vispy", "vispy")
_ensure_pkg("sounddevice", "sounddevice")

from PyQt5 import QtWidgets, QtCore
from vispy import app, scene
import sounddevice as sd

app.use_app("pyqt5")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# -----------------------------------------------------------------------------#
# Musical helpers & preset scales
# -----------------------------------------------------------------------------#
def hz_to_semitones(f, f0=110.0):
    return 12.0 * np.log2(max(1e-9, float(f)) / float(f0))

def build_scale_from_classes(classes, ensure_zero=True, max_len=7):
    uniq = sorted(set(int(c) % 12 for c in classes))
    if ensure_zero and 0 not in uniq:
        uniq = [0] + [x for x in uniq if x != 0]
    if len(uniq) > max_len:
        out = [uniq[0]]
        for u in uniq[1:]:
            if len(out) >= max_len: break
            if min(((u - v) % 12 for v in out)) >= 2:
                out.append(u)
        uniq = sorted(out)
    return uniq

PRESET_SCALES = {
    "S2-derived": None,  # filled at runtime
    "Hirajoshi":   [0, 2, 3, 7, 8],
    "Pelog-ish":   [0, 1, 3, 7, 8],
    "Messiaen-3":  [0, 2, 3, 4, 6, 7, 8, 10],
    "Bhairav-ish": [0, 1, 4, 5, 7, 8, 11],
}

def quantize_to_scale(semi, scale):
    o = int(np.floor(semi / 12.0))
    pc = semi - 12 * o
    best = min(scale, key=lambda s: abs(s - pc))
    return 12 * o + best

# Simple HSV→RGB for colouring by pitch class
def hsv_to_rgb(h, s, v):
    h = float(h % 1.0); s = float(np.clip(s, 0, 1)); v = float(np.clip(v, 0, 1))
    i = int(h * 6.0); f = h * 6.0 - i
    p = v * (1.0 - s); q = v * (1.0 - s * f); t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    if i == 0: r, g, b = v, t, p
    elif i == 1: r, g, b = q, v, p
    elif i == 2: r, g, b = p, v, t
    elif i == 3: r, g, b = p, q, v
    elif i == 4: r, g, b = t, p, v
    else: r, g, b = v, p, q
    return (float(r), float(g), float(b))

# -----------------------------------------------------------------------------#
# Data structures shared with the tri-harmonic core
# -----------------------------------------------------------------------------#
class SphereFunction(Enum):
    TEMPORAL = "temporal_memory"
    SEMANTIC = "semantic_drift"
    HARMONIC = "harmonic_feedback"

@dataclass
class DataPoint:
    value: float          # frequency (Hz) or value
    timestamp: float
    symbol: str
    phase: float          # routing phase (0..2π)
    spectrum: Optional[np.ndarray]
    energy: float         # 0..1
    duration: float
    metadata: Dict[str, Any] = None
    def clone(self):
        return DataPoint(
            value=self.value, timestamp=self.timestamp, symbol=self.symbol,
            phase=self.phase, spectrum=self.spectrum.copy() if self.spectrum is not None else None,
            energy=self.energy, duration=self.duration,
            metadata=self.metadata.copy() if self.metadata else {}
        )

# -----------------------------------------------------------------------------#
# Tri-Harmonic core (routing + visuals plumbing)
# -----------------------------------------------------------------------------#
class SymbolicSphere:
    def __init__(self, radius, plane, function, capacity=100, decay_lambda=0.95, rotation_speed=0.5):
        self.radius = radius; self.plane = plane; self.function = function
        self.capacity = capacity; self.decay_lambda = decay_lambda; self.rotation_speed = rotation_speed
        self.memory_ring = deque(maxlen=capacity)
        self.phase_offset = np.random.random() * 2 * np.pi
        self.symbol_set = set()
        self.output_queue = deque()
        self.total_residence_time = 0.0; self.exit_count = 0

    def _pos(self, phase):
        a = phase + self.phase_offset
        if self.plane == "XY": return (self.radius*np.cos(a), self.radius*np.sin(a), 0.0)
        if self.plane == "YZ": return (0.0, self.radius*np.cos(a), self.radius*np.sin(a))
        if self.plane == "XZ": return (self.radius*np.cos(a), 0.0, self.radius*np.sin(a))
        return (0.0, 0.0, 0.0)

    def update_rotation(self):
        self.phase_offset = (self.phase_offset + self.rotation_speed) % (2*np.pi)
        for dp in self.memory_ring: dp.energy *= self.decay_lambda

    def can_accept(self, dp: DataPoint):
        s = 0.2 + 0.6*np.clip(dp.energy,0,1)
        if dp.symbol in self.symbol_set: s += 0.2
        elif len(self.symbol_set) < 20: s += 0.1
        return (s > 0.25, s)

    def inject(self, dp: DataPoint):
        d = dp.clone(); d.metadata = d.metadata or {}
        d.metadata.update({"sphere_position": self._pos(d.phase), "entry_time": time.time(),
                           "sphere_plane": self.plane, "sphere_function": self.function.value})
        self.memory_ring.append(d); self.symbol_set.add(d.symbol)

    def extract_ready(self):
        ready, rem = [], deque(); now = time.time()
        for d in self.memory_ring:
            dt = now - d.metadata.get("entry_time", now)
            exit_cond = (dt > 1.2) or (d.energy < 0.05) or (len(self.memory_ring) > self.capacity-2)
            if exit_cond:
                d.metadata["exit_reason"] = "timeout"; d.metadata["residence_time"] = dt
                ready.append(d); self.total_residence_time += dt; self.exit_count += 1
            else: rem.append(d)
        self.memory_ring = rem
        return ready

    def apply_transformations(self):
        for d in self.memory_ring:
            d.phase = (d.phase + self.rotation_speed) % (2*np.pi)
            d.metadata["sphere_position"] = self._pos(d.phase)

class ResonanceGate:
    def __init__(self, phase_tolerance=0.6, energy_threshold=0.08):
        self.phase_tolerance = phase_tolerance; self.energy_threshold = energy_threshold
    def check_resonance(self, dp: DataPoint, sphere: SymbolicSphere):
        if dp.energy < self.energy_threshold: return (False, 0.0)
        pd = abs(dp.phase - sphere.phase_offset) % (2*np.pi)
        align = 1.0 - min(pd, 2*np.pi-pd)/max(1e-6, self.phase_tolerance)
        s = 0.3*np.clip(align, 0, 1)
        ok, sc = sphere.can_accept(dp); s += (0.5 if ok else 0.2)*sc
        return (s > 0.25, s)

class TriHarmonicCore:
    def __init__(self, major_radius=10.0, minor_radius=3.0, enable_s2=True, enable_s3=True):
        self.major_radius = major_radius; self.minor_radius = minor_radius
        r = minor_radius * 0.8
        self.sphere_s1 = SymbolicSphere(r, "XY", SphereFunction.TEMPORAL, rotation_speed=0.45)
        self.sphere_s2 = SymbolicSphere(r, "YZ", SphereFunction.SEMANTIC, rotation_speed=0.33, decay_lambda=0.98) if enable_s2 else None
        self.sphere_s3 = SymbolicSphere(r, "XZ", SphereFunction.HARMONIC, rotation_speed=0.7,  decay_lambda=0.92) if enable_s3 else None
        self.spheres = [self.sphere_s1] + ([self.sphere_s2] if self.sphere_s2 else []) + ([self.sphere_s3] if self.sphere_s3 else [])
        self.gate = ResonanceGate()
        self.torus_buffer = deque(maxlen=1000)
        self.metrics = {"total_processed":0,"s1_entries":0,"s1_exits":0,"s2_entries":0,"s2_exits":0,"s3_entries":0,"s3_exits":0,
                        "torus_direct":0,"total_resonance_checks":0,"avg_resonance_score":0.0}
        self.resonance_scores: List[float] = []

    def _torus_pos(self, phase: float, energy: float):
        R, rr = self.major_radius, self.minor_radius * 0.9
        u = phase; v = energy * 2 * np.pi
        x = (R + rr*np.cos(v))*np.cos(u); y = (R + rr*np.cos(v))*np.sin(u); z = rr*np.sin(v)
        return [float(x), float(y), float(z)]

    def process(self, dp: DataPoint):
        self.metrics["total_processed"] += 1
        best, best_sc = None, 0.0
        for s in self.spheres:
            ok, sc = self.gate.check_resonance(dp, s)
            self.metrics["total_resonance_checks"] += 1; self.resonance_scores.append(sc)
            if ok and sc > best_sc: best, best_sc = s, sc
        if best:
            best.inject(dp); key = {self.sphere_s1:"s1_entries", self.sphere_s2:"s2_entries", self.sphere_s3:"s3_entries"}[best]
            self.metrics[key] += 1
        else:
            self.torus_buffer.append(dp); self.metrics["torus_direct"] += 1

        for s in self.spheres:
            s.update_rotation(); s.apply_transformations()
            for ed in s.extract_ready():
                ed.metadata["sphere_processed"] = True
                self.torus_buffer.append(ed)
                k = {self.sphere_s1:"s1_exits", self.sphere_s2:"s2_exits", self.sphere_s3:"s3_exits"}[s]
                self.metrics[k] += 1

        if self.resonance_scores:
            self.metrics["avg_resance"] = float(np.mean(self.resonance_scores[-100:]))

# -----------------------------------------------------------------------------#
# Enhanced Melodic Sphere Engine
# -----------------------------------------------------------------------------#
class MelodicSphereEngine:
    def __init__(self, audio_engine, tri_harmonic_core):
        self.audio = audio_engine
        self.core = tri_harmonic_core
        
        # Melodic state tracking
        self.sphere_melody_influence = 0.7
        self.harmonic_memory_depth = 8
        self.melodic_tension_tracking = True
        self.adaptive_rhythm_from_spheres = True
        
        # Sphere-driven melodic parameters
        self.s1_melody_weight = 0.4  # Temporal memory influences rhythm
        self.s2_melody_weight = 0.8  # Semantic drift drives melody
        self.s3_melody_weight = 0.6  # Harmonic feedback affects harmony
        
        # Advanced melodic features
        self.phrase_memory = []
        self.tension_curve = 0.0
        self.melodic_momentum = np.array([0.0, 0.0])  # [bass, lead] momentum
        
    def extract_sphere_melodic_data(self):
        """Extract musical information from sphere states"""
        melodic_data = {
            'temporal_energy': 0.0,
            'semantic_drift': 0.0,
            'harmonic_tension': 0.0,
            'phase_relationships': [],
            'energy_distribution': [],
            'symbol_density': 0.0
        }
        
        # S1 (Temporal) - rhythm and timing variations
        if self.core.sphere_s1.memory_ring:
            energies = [dp.energy for dp in self.core.sphere_s1.memory_ring]
            phases = [dp.phase for dp in self.core.sphere_s1.memory_ring]
            melodic_data['temporal_energy'] = np.mean(energies)
            melodic_data['phase_relationships'] = self._analyze_phase_patterns(phases)
            
        # S2 (Semantic) - melodic direction and intervals
        if self.core.sphere_s2 and self.core.sphere_s2.memory_ring:
            values = [dp.value for dp in self.core.sphere_s2.memory_ring]
            melodic_data['semantic_drift'] = self._calculate_melodic_drift(values)
            melodic_data['symbol_density'] = len(self.core.sphere_s2.symbol_set) / max(1, len(self.core.sphere_s2.memory_ring))
            
        # S3 (Harmonic) - chord progressions and harmonic rhythm
        if self.core.sphere_s3 and self.core.sphere_s3.memory_ring:
            recent_harmonics = [dp.value for dp in list(self.core.sphere_s3.memory_ring)[-4:]]
            melodic_data['harmonic_tension'] = self._calculate_harmonic_tension(recent_harmonics)
            
        return melodic_data
    
    def _analyze_phase_patterns(self, phases):
        """Analyze rhythmic patterns from phase relationships"""
        if len(phases) < 2:
            return {'regularity': 0.5, 'acceleration': 0.0}
            
        # Calculate phase differences for rhythm analysis
        diffs = np.diff(phases)
        regularity = 1.0 - np.std(diffs) / (np.pi + 1e-6)  # More regular = higher value
        acceleration = np.mean(np.diff(diffs)) if len(diffs) > 1 else 0.0
        
        return {
            'regularity': np.clip(regularity, 0.0, 1.0),
            'acceleration': np.clip(acceleration, -1.0, 1.0)
        }
    
    def _calculate_melodic_drift(self, frequencies):
        """Calculate the overall melodic direction and momentum"""
        if len(frequencies) < 3:
            return 0.0
            
        # Convert to semitones for musical analysis
        semitones = [hz_to_semitones(f, self.audio.root) for f in frequencies]
        
        # Calculate overall direction (upward/downward motion)
        direction = np.sum(np.diff(semitones[-6:]))  # Last 6 notes
        return np.tanh(direction / 12.0)  # Normalize to [-1, 1]
    
    def _calculate_harmonic_tension(self, frequencies):
        """Calculate harmonic tension from recent frequencies"""
        if len(frequencies) < 2:
            return 0.0
            
        # Convert to pitch classes and analyze dissonance
        pitch_classes = [int(hz_to_semitones(f, self.audio.root)) % 12 for f in frequencies]
        
        # Simple dissonance calculation based on intervals
        tension = 0.0
        for i in range(len(pitch_classes) - 1):
            interval = abs(pitch_classes[i] - pitch_classes[i+1]) % 12
            # Dissonant intervals: minor 2nd, major 7th, tritone
            if interval in [1, 6, 11]:
                tension += 0.8
            elif interval in [2, 10]:  # Major 2nd, minor 7th
                tension += 0.4
                
        return np.tanh(tension / len(pitch_classes))
    
    def generate_sphere_influenced_melody(self, step_idx, current_scale):
        """Generate melody modifications based on sphere data"""
        melodic_data = self.extract_sphere_melodic_data()
        
        # Base patterns (keep existing)
        bass_note = self.audio.bass_pat[step_idx % len(self.audio.bass_pat)]
        lead_note = self.audio.lead_pat[step_idx % len(self.audio.lead_pat)]
        
        # Sphere-influenced modifications
        modifications = {
            'bass_offset': 0,
            'lead_offset': 0,
            'bass_octave': 0,
            'lead_octave': 0,
            'add_passing_tones': False,
            'rhythmic_displacement': 0.0,
            'micro_timing': 0.0,
            'energy_boost': 0.0,
            'filter_mod': 0.0
        }
        
        # Apply sphere influence scaling
        influence = self.sphere_melody_influence
        if influence < 0.1:
            return modifications
        
        # S2 (Semantic) drives melodic contour
        if melodic_data['semantic_drift'] != 0.0:
            drift = melodic_data['semantic_drift'] * self.s2_melody_weight * influence
            
            # Push melody in direction of semantic drift
            if abs(drift) > 0.3:
                lead_push = int(np.sign(drift) * np.ceil(abs(drift) * 3))
                modifications['lead_offset'] = lead_push
                
            # Add passing tones when semantic density is high
            if melodic_data['symbol_density'] > 0.6:
                modifications['add_passing_tones'] = True
        
        # S1 (Temporal) affects rhythm and micro-timing
        phase_data = melodic_data['phase_relationships']
        if phase_data and self.s1_melody_weight > 0.1:
            temporal_influence = self.s1_melody_weight * influence
            
            # Irregular phases create syncopation
            if phase_data['regularity'] < 0.4:
                modifications['rhythmic_displacement'] = (0.5 - phase_data['regularity']) * 0.3 * temporal_influence
                
            # Acceleration affects micro-timing
            modifications['micro_timing'] = phase_data.get('acceleration', 0.0) * 0.1 * temporal_influence
            
            # Energy affects dynamics
            modifications['energy_boost'] = melodic_data['temporal_energy'] * 0.3 * temporal_influence
        
        # S3 (Harmonic) influences bass line and octave choices
        tension = melodic_data['harmonic_tension']
        if tension > 0.0 and self.s3_melody_weight > 0.1:
            harmonic_influence = self.s3_melody_weight * influence
            
            if tension > 0.5:
                # High tension: wider intervals, octave jumps
                modifications['bass_octave'] = 1 if step_idx % 4 == 0 else 0
                modifications['lead_octave'] = -1 if tension > 0.7 else 0
                modifications['filter_mod'] = tension * harmonic_influence
            elif tension < 0.2:
                # Low tension: stepwise motion, stay in comfortable range
                modifications['bass_offset'] = np.clip(modifications['bass_offset'], -1, 1)
                modifications['lead_offset'] = np.clip(modifications['lead_offset'], -2, 2)
                modifications['filter_mod'] = -0.3 * harmonic_influence
        
        return modifications
    
    def apply_melodic_modifications(self, step_idx, modifications):
        """Apply the calculated modifications to the current step"""
        
        # Get base notes
        bass_idx = step_idx % len(self.audio.bass_pat)
        lead_idx = step_idx % len(self.audio.lead_pat)
        
        bass_base = self.audio.bass_pat[bass_idx]
        lead_base = self.audio.lead_pat[lead_idx]
        
        # Apply offsets with scale constraint
        scale_len = len(self.audio.scale)
        
        new_bass = (bass_base + modifications['bass_offset']) % scale_len
        new_lead = (lead_base + modifications['lead_offset']) % scale_len
        
        # Apply octave shifts
        bass_oct = modifications['bass_octave']
        lead_oct = modifications['lead_octave']
        
        # Calculate frequencies
        bass_semi = self.audio.scale[new_bass] + bass_oct * 12
        lead_semi = self.audio.scale[new_lead] + lead_oct * 12
        
        self.audio.b_freq = self.audio.root * (2 ** (bass_semi / 12.0))
        self.audio.l_freq = self.audio.root * (2 ** (lead_semi / 12.0))
        
        # Store modifications for use in audio callback
        self.audio._micro_timing_offset = modifications.get('micro_timing', 0.0)
        self.audio._energy_boost = modifications.get('energy_boost', 0.0)
        self.audio._filter_mod = modifications.get('filter_mod', 0.0)
        
        # Add passing tones (could trigger additional shorter notes)
        if modifications.get('add_passing_tones', False):
            self._schedule_passing_tone(step_idx)
    
    def _schedule_passing_tone(self, step_idx):
        """Schedule a passing tone between current and next note"""
        # For now, we can modify the current note slightly
        current_semi = hz_to_semitones(self.audio.l_freq, self.audio.root)
        
        # Add a small pitch bend or quick grace note effect
        # This could be implemented as a brief frequency modulation
        pass
    
    def update_melodic_memory(self, bass_freq, lead_freq, step_idx):
        """Update melodic memory for phrase analysis"""
        phrase_data = {
            'step': step_idx,
            'bass': bass_freq,
            'lead': lead_freq,
            'timestamp': time.time(),
            'sphere_state': self.extract_sphere_melodic_data()
        }
        
        self.phrase_memory.append(phrase_data)
        
        # Keep only recent history
        if len(self.phrase_memory) > 32:
            self.phrase_memory.pop(0)
        
        # Update tension curve
        self._update_tension_curve()
    
    def _update_tension_curve(self):
        """Calculate overall musical tension from phrase memory"""
        if len(self.phrase_memory) < 4:
            return
            
        recent = self.phrase_memory[-8:]
        
        # Analyze melodic motion, harmonic content, rhythmic activity
        lead_motion = np.diff([hz_to_semitones(p['lead'], self.audio.root) for p in recent])
        
        # Large intervals increase tension
        interval_tension = np.sum(np.abs(lead_motion)) / len(lead_motion)
        
        # Sphere activity increases tension
        sphere_tension = np.mean([p['sphere_state']['harmonic_tension'] for p in recent])
        
        self.tension_curve = np.clip(interval_tension * 0.1 + sphere_tension, 0.0, 1.0)

# -----------------------------------------------------------------------------#
# Berlin audio engine (with waveforms + smoothed delay + melodic integration)
# -----------------------------------------------------------------------------#
class ParamSmoother:
    def __init__(self, sr, ramp_ms=25.0):
        self.sr = sr; self.cur = 0.0; self.target = 0.0; self.set_time(ramp_ms)
    def set_time(self, ramp_ms):
        tau = max(1e-3, ramp_ms/1000.0)
        self.alpha = 1.0 - np.exp(-1.0/(self.sr*tau))
    def set_target(self, v): self.target = float(v)
    def next_block(self, n):
        out = np.empty(n, dtype=np.float32); c = self.cur; a = self.alpha; tgt = self.target
        for i in range(n): c += (tgt - c) * a; out[i] = c
        self.cur = c; return out

def tri_from_sine(phase):
    return (2.0/np.pi)*np.arcsin(np.clip(np.sin(2*np.pi*phase), -1, 1))

def osc_block(shape, freq, n, sr, phase0=0.0, pwm=0.5, detune_cents=7.0):
    t = (phase0 + np.arange(n)/sr*freq) % 1.0
    if shape == "sine":
        out = np.sin(2*np.pi*t, dtype=np.float32)
    elif shape == "triangle":
        out = tri_from_sine(t).astype(np.float32)
    elif shape == "square":
        out = np.sign(np.sin(2*np.pi*t)).astype(np.float32)
    elif shape == "pulse":
        duty = float(np.clip(pwm, 0.05, 0.95))
        out = ((t % 1.0) < duty).astype(np.float32)*2 - 1
    elif shape == "supersaw":
        det = 2**(detune_cents/1200.0)
        t2 = (phase0 + np.arange(n)/sr*(freq*det)) % 1.0
        saw1 = (2.0*t - 1.0); saw2 = (2.0*t2 - 1.0)
        out = 0.5*(saw1 + saw2).astype(np.float32)
    else:  # "saw"
        out = (2.0*t - 1.0).astype(np.float32)
    phase1 = (phase0 + n*freq/sr) % 1.0
    return out, phase1

class OnePoleLP:
    def __init__(self, sr, cutoff=800.0):
        self.sr = sr; self.z = 0.0; self.set_cutoff(cutoff)
    def set_cutoff(self, hz):
        hz = max(20.0, min(self.sr*0.45, hz))
        x = np.exp(-2*np.pi*hz/self.sr); self.a = 1.0 - x; self.b = x
    def process(self, x):
        y = np.empty_like(x); z = self.z; a=self.a; b=self.b
        for i in range(x.size): z = a*x[i] + b*z; y[i] = z
        self.z = z; return y

@dataclass
class Voice:
    shape: str = "saw"
    phase: float = 0.0
    pwm: float = 0.5
    detune_cents: float = 7.0
    filt: OnePoleLP = None

class BerlinAudio:
    def __init__(self, sr=44100):
        self.sr = sr; self._lock = threading.Lock()
        # Global
        self.master = 0.8; self.bpm = 120.0; self.swing = 0.0
        self.steps = 16; self.sample_counter = 0; self.step_samples = self._calc_step_samples()
        self.next_step_at = self.step_samples; self.step_idx = 0
        self.time = 0.0
        # Delay
        self.delay_buf = np.zeros(int(sr*2.5), dtype=np.float32); self.dw = 0
        self.delay_ms = 350.0; self.delay_fb = 0.35; self.delay_mix = 0.25
        self._mix_s = ParamSmoother(sr, 25); self._fb_s = ParamSmoother(sr, 25); self._dt_s = ParamSmoother(sr, 60)
        self._mix_s.set_target(self.delay_mix); self._fb_s.set_target(self.delay_fb); self._dt_s.set_target(self.delay_ms/1000.0)
        # Vibrato
        self.vib_cents = 4.0; self.vib_rate = 5.5
        # Voices
        self.bass = Voice(shape="saw", filt=OnePoleLP(sr, 220.0))
        self.lead = Voice(shape="square", filt=OnePoleLP(sr, 1800.0))
        self.drive = 1.2
        # Counterpoint
        self.cp_enabled = False
        self.cp_level = 0.18
        self.cp_lag_steps = 2
        self.cp_invert = True
        self.cp_scale_name = "S2-derived"
        self._get_s2_scale = None
        self.cp = Voice(shape="sine", filt=OnePoleLP(sr, 2400.0))
        self.cp_pat = [0, 2, 4, 6, 4, 3, 2, 1, 0, 2, 4, 5, 4, 2, 1, 0]
        self._lead_step_hist: List[float] = []
        # Drums env
        self.k_env = 0.0; self.k_phase = 0.0; self.s_env = 0.0; self.h_env = 0.0
        # Patterns (minor pentatonic around A)
        self.root = 110.0
        self.scale = np.array([0, 3, 5, 7, 10, 12])
        self.bass_pat = [0,0,3,0, 5,5,3,0, 0,7,5,3, 10,7,5,3]
        self.lead_pat = [12,15,12,17, 12,19,17,15, 12,15,12,17, 19,17,15,12]
        self.k_pat =   [1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0]
        self.s_pat =   [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0]
        self.h_pat =   [1,0,1,0, 1,0,1,0, 1,0,1,0, 1,0,1,0]
        # Targets per step
        self.b_freq = self.root; self.l_freq = self.root*2
        self.b_gain = 0.25; self.l_gain = 0.22
        
        # Melodic engine enhancement attributes
        self._micro_timing_offset = 0.0
        self._energy_boost = 0.0
        self._filter_mod = 0.0
        self.melodic_engine = None  # Will be set later
        
        # Event callback (set by UI later)
        self.on_step_event = None

        self.stream = sd.OutputStream(samplerate=sr, channels=1, dtype="float32", blocksize=512, callback=self._cb)
        self.stream.start()

    def set_melodic_engine(self, engine):
        """Set the melodic engine after initialization"""
        with self._lock:
            self.melodic_engine = engine

    # Setters
    def _calc_step_samples(self):
        spb = 60.0 / max(1.0, self.bpm); return int((spb/4.0) * self.sr)

    def set_bpm(self, bpm):
        with self._lock:
            self.bpm = float(np.clip(bpm, 20, 300)); self.step_samples = self._calc_step_samples()
            self.sample_counter = 0; self.next_step_at = self.step_samples; self.step_idx = 0
            self._lead_step_hist.clear()

    def set_swing(self, pct):  # 0..0.5
        with self._lock: self.swing = float(np.clip(pct, 0.0, 0.5))

    def set_voice(self, which: str, shape: str=None, pwm: float=None, detune: float=None, cutoff: float=None):
        v = self.bass if which=="bass" else self.lead
        with self._lock:
            if shape is not None: v.shape = shape
            if pwm   is not None: v.pwm = float(np.clip(pwm, 0.05, 0.95))
            if detune is not None: v.detune_cents = float(np.clip(detune, 0.0, 25.0))
            if cutoff is not None: v.filt.set_cutoff(float(np.clip(cutoff, 50, 8000)))

    def set_drive(self, drive_x):  # 0.5 .. 3.0
        with self._lock: self.drive = float(np.clip(drive_x, 0.5, 3.0))

    def set_delay(self, ms, fb, mix):
        with self._lock:
            self.delay_ms = float(np.clip(ms, 1.0, 1500.0))
            self.delay_fb = float(np.clip(fb, 0.0, 0.95))
            self.delay_mix = float(np.clip(mix, 0.0, 1.0))
            self._mix_s.set_target(self.delay_mix); self._fb_s.set_target(self.delay_fb); self._dt_s.set_target(self.delay_ms/1000.0)

    def set_vibrato(self, cents, rate_hz):
        with self._lock:
            self.vib_cents = float(np.clip(cents, 0, 50)); self.vib_rate = float(np.clip(rate_hz, 0.1, 12.0))

    def randomize_patterns(self):
        rng = np.random.default_rng()
        self.bass_pat = [int(rng.choice([0,0,3,5,7,10])) for _ in range(16)]
        self.lead_pat = [int(rng.choice([12,15,17,19,24,27])) for _ in range(16)]
        self.k_pat = [1 if i%4==0 else 0 for i in range(16)]
        self.s_pat = [1 if i in (4,12) else 0 for i in range(16)]
        self.h_pat = [1 if i%2==0 else 0 for i in range(16)]

    def set_counterpoint(self, enabled=None, scale_name=None, lag_steps=None, invert=None, level=None):
        with self._lock:
            if enabled  is not None: self.cp_enabled = bool(enabled)
            if scale_name is not None: self.cp_scale_name = str(scale_name)
            if lag_steps is not None: self.cp_lag_steps = int(np.clip(lag_steps, 0, 32))
            if invert is not None: self.cp_invert = bool(invert)
            if level is not None: self.cp_level = float(np.clip(level, 0.0, 1.0))

    def set_cp_voice(self, shape=None, cutoff=None, detune=None, pwm=None):
        with self._lock:
            if shape  is not None: self.cp.shape = shape
            if cutoff is not None: self.cp.filt.set_cutoff(float(np.clip(cutoff, 200, 12000)))
            if detune is not None: self.cp.detune_cents = float(np.clip(detune, 0.0, 25.0))
            if pwm    is not None: self.cp.pwm = float(np.clip(pwm, 0.05, 0.95))

    def bind_s2_scale_provider(self, fn):
        self._get_s2_scale = fn

    # --- Always return a safe, non-empty scale list
    def _resolve_cp_scale(self, name):
        sc = None
        if name == "S2-derived" and self._get_s2_scale:
            try:
                sc = self._get_s2_scale()
            except Exception:
                sc = None
        if not sc:
            sc = PRESET_SCALES.get(name)
        if not sc:
            sc = PRESET_SCALES["Hirajoshi"]
        return list(sc)

    # Enhanced step advance with melodic engine integration
    def _advance_step(self):
        i = self.step_idx % self.steps
        
        # Apply melodic engine if available
        if self.melodic_engine and self.melodic_engine.sphere_melody_influence > 0.1:
            # Get sphere-influenced modifications
            modifications = self.melodic_engine.generate_sphere_influenced_melody(i, self.scale)
            
            # Apply modifications
            self.melodic_engine.apply_melodic_modifications(i, modifications)
            
            # Update dynamics based on sphere data
            melodic_data = self.melodic_engine.extract_sphere_melodic_data()
            
            # Sphere energy affects gain
            energy_boost = getattr(self, '_energy_boost', 0.0)
            self.b_gain = 0.28 + energy_boost
            self.l_gain = 0.24 + energy_boost
            
            # Harmonic tension affects filter cutoffs
            filter_mod = getattr(self, '_filter_mod', 0.0)
            if filter_mod > 0.3:
                # High tension: brighter sound
                new_cutoff = min(8000.0, 1800.0 + filter_mod * 2000.0)
                self.lead.filt.set_cutoff(new_cutoff)
            elif filter_mod < -0.2:
                # Low tension: warmer sound
                new_cutoff = max(400.0, 1800.0 + filter_mod * 1000.0)
                self.lead.filt.set_cutoff(new_cutoff)
            
            # Update melodic memory
            self.melodic_engine.update_melodic_memory(self.b_freq, self.l_freq, self.step_idx)
            
        else:
            # Original behavior when melodic engine is disabled
            semi_b = self.scale[0] + self.bass_pat[i]; self.b_freq = self.root * (2**(semi_b/12.0))
            semi_l = self.lead_pat[i];                  self.l_freq = self.root * (2**(semi_l/12.0))
            self.b_gain = 0.28; self.l_gain = 0.24

        # Continue with original drum triggers
        if self.k_pat[i]: self.k_env = 1.0; self.k_phase = 0.0
        if self.s_pat[i]: self.s_env = 1.0
        if self.h_pat[i]: self.h_env = 1.0

        self._lead_step_hist.append(self.l_freq)
        if len(self._lead_step_hist) > 256: self._lead_step_hist.pop(0)

        if self.on_step_event:
            if self.k_pat[i]: self.on_step_event("kick", 55.0, 1.0, i)
            self.on_step_event("bass", self.b_freq, 0.8, i)
            self.on_step_event("lead", self.l_freq, 0.6, i)

        # Apply step timing with micro-timing
        base = self._calc_step_samples()
        if (i%2)==1: base = int(base * (1.0 + self.swing))
        
        # Apply micro-timing from sphere data
        micro_offset = getattr(self, '_micro_timing_offset', 0.0)
        base = int(base * (1.0 + micro_offset))
        
        self.next_step_at += base; self.step_idx += 1

    # Drums
    def _kick(self, n):
        if self.k_env <= 1e-4: return np.zeros(n, dtype=np.float32)
        t = np.arange(n)/self.sr
        f0, f1, dtime = 90.0, 35.0, 0.05
        f = f1 + (f0-f1)*np.exp(-t/dtime)
        self.k_env *= np.exp(-n/(self.sr*0.22))
        ph = self.k_phase + 2*np.pi*np.cumsum(f)/self.sr
        self.k_phase = ph[-1] % (2*np.pi)
        return (np.sin(ph) * self.k_env * 0.95).astype(np.float32)

    def _snare(self, n):
        if self.s_env <= 1e-4: return np.zeros(n, dtype=np.float32)
        noise = (np.random.rand(n).astype(np.float32)*2 - 1)
        d = noise - np.concatenate(([0.0], noise[:-1])) * 0.85
        self.s_env *= np.exp(-n/(self.sr*0.14))
        return d * self.s_env * 0.5

    def _hihat(self, n):
        if self.h_env <= 1e-4: return np.zeros(n, dtype=np.float32)
        noise = (np.random.rand(n).astype(np.float32)*2 - 1)
        y = noise - np.concatenate(([0.0], noise[:-1])) * 0.5
        self.h_env *= np.exp(-n/(self.sr*0.045))
        return y * self.h_env * 0.22

    # Delay with smoothing + fractional interp
    def _delay(self, x):
        n = x.size; out = np.empty_like(x)
        mix = self._mix_s.next_block(n); fb = self._fb_s.next_block(n); dts = self._dt_s.next_block(n)
        dSamp = np.clip(dts*self.sr, 1.0, self.delay_buf.size-2)
        for i in range(n):
            rd_float = (self.dw - dSamp[i]) % self.delay_buf.size
            i0 = int(rd_float); frac = rd_float - i0; i1 = (i0+1) % self.delay_buf.size
            y = (1.0-frac)*self.delay_buf[i0] + frac*self.delay_buf[i1]
            out[i] = x[i]*(1.0 - mix[i]) + y*mix[i]
            self.delay_buf[self.dw] = x[i] + y*fb[i]
            self.dw = (self.dw + 1) % self.delay_buf.size
        return out

    # Audio callback
    def _cb(self, out, frames, time_info, status):
        with self._lock:
            master = self.master; drive = self.drive
            b_shape, l_shape = self.bass.shape, self.lead.shape
            pwm_b, pwm_l = self.bass.pwm, self.lead.pwm
            det_b, det_l = self.bass.detune_cents, self.lead.detune_cents
            cp_enabled = self.cp_enabled
            cp_scale_name = self.cp_scale_name
            cp_lag_steps = self.cp_lag_steps
            cp_invert = self.cp_invert
            cp_level = self.cp_level

        start = self.sample_counter; end = start + frames
        while self.next_step_at <= end: self._advance_step()
        self.sample_counter = end

        b_raw, self.bass.phase = osc_block(b_shape, self.b_freq, frames, self.sr, self.bass.phase, pwm_b, det_b)
        l_raw, self.lead.phase = osc_block(l_shape, self.l_freq, frames, self.sr, self.lead.phase, pwm_l, det_l)
        b_raw *= self.b_gain; l_raw *= self.l_gain
        b_raw = np.tanh(drive*b_raw); l_raw = np.tanh(drive*l_raw)
        b = self.bass.filt.process(b_raw); l = self.lead.filt.process(l_raw)

        if cp_enabled:
            scale = self._resolve_cp_scale(cp_scale_name)
            scale_len = max(1, len(scale))
            if self._lead_step_hist:
                idx = max(0, len(self._lead_step_hist) - 1 - cp_lag_steps)
                lead_lag_hz = self._lead_step_hist[idx]
            else:
                lead_lag_hz = self.l_freq
            lead_semi_lag = hz_to_semitones(lead_lag_hz, self.root)
            i = (self.step_idx % len(self.cp_pat))
            deg = self.cp_pat[i] % scale_len
            target_pc = float(scale[deg])
            anchor = 0.0 if not cp_invert else (lead_semi_lag % 12.0)
            def reflect(pc, a): return (2*a - pc) % 12.0
            pc = reflect(target_pc, anchor) if cp_invert else target_pc
            cp_semi = quantize_to_scale(12*np.floor(lead_semi_lag/12.0) + pc, scale)
            cp_freq = self.root * (2.0 ** (cp_semi/12.0))
            if cp_invert:
                cp_freq *= (1.005 if cp_freq > self.l_freq else 0.995)
            cp_raw, self.cp.phase = osc_block(self.cp.shape, cp_freq, frames, self.sr, self.cp.phase, self.cp.pwm, self.cp.detune_cents)
            cp_raw *= cp_level
            cp_raw = np.tanh(drive*cp_raw*0.9)
            cp = self.cp.filt.process(cp_raw)
        else:
            cp = 0.0

        k = self._kick(frames); s = self._snare(frames); h = self._hihat(frames)
        mix = b + l + (cp if isinstance(cp, np.ndarray) else 0.0) + k + s + h

        if self.delay_mix > 0.001: mix = self._delay(mix)

        mix *= self.master; np.clip(mix, -0.98, 0.98, out=mix)
        out[:,0] = mix; self.time += frames/self.sr

    def stop(self): self.stream.stop(); self.stream.close()

# -----------------------------------------------------------------------------#
# Visualizer with torus + spheres + controls + 3D pooled cymatic pulse rings + melodic controls
# -----------------------------------------------------------------------------#
class BerlinTriVis:
    def __init__(self):
        # Core + Audio
        self.core = TriHarmonicCore(enable_s2=True, enable_s3=True)
        self.audio = BerlinAudio()
        
        # Initialize melodic engine and link it
        self.melodic_engine = MelodicSphereEngine(self.audio, self.core)
        self.audio.set_melodic_engine(self.melodic_engine)

        # Qt/VisPy shell
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        self.win = QtWidgets.QWidget(); self.win.setWindowTitle("Berlin × Tri-Harmonic (Enhanced Melodic + Sequencer + 3D Cymatic Pulses + Counterpoint)")

        root = QtWidgets.QHBoxLayout(self.win); root.setContentsMargins(8,8,8,8); root.setSpacing(8)
        left = QtWidgets.QVBoxLayout(); right = QtWidgets.QVBoxLayout(); root.addLayout(left, 1); root.addLayout(right, 0)

        # Canvas 3D
        self.canvas = scene.SceneCanvas(title="TRI-HARMONIC VIS", size=(1280, 900), bgcolor="black")
        left.addWidget(self.canvas.native, stretch=1)
        grid = self.canvas.central_widget.add_grid()
        self.view3d = grid.add_view(row=0, col=0, camera="turntable")
        self.view3d.camera.distance = 40; self.view3d.camera.elevation = 30; self.view3d.camera.azimuth = 45; self.view3d.camera.fov = 60

        # Geometry
        self._create_torus_wireframe(); self._create_sphere_wireframes()

        # Particles
        self.torus_particles = scene.visuals.Markers(parent=self.view3d.scene)
        self.s1_particles = scene.visuals.Markers(parent=self.view3d.scene)
        self.s2_particles = scene.visuals.Markers(parent=self.view3d.scene)
        self.s3_particles = scene.visuals.Markers(parent=self.view3d.scene)
        for p in (self.torus_particles, self.s1_particles, self.s2_particles, self.s3_particles):
            p.set_gl_state("translucent", depth_test=True)

        # ------------ Pooled cymatic pulse rings (init BEFORE wiring callback) ------------ #
        self.pulse_group = scene.Node(parent=self.view3d.scene)
        self.max_pulse_rings = 24
        self.pulse_lifetime = 3.0
        self.cym_pulses_enabled = True
        self._pulse_theta = np.linspace(0, 2*np.pi, 128).astype(np.float32)
        self._pulse_pool = []
        for _ in range(self.max_pulse_rings):
            line = scene.visuals.Line(np.zeros((0,3), dtype=np.float32),
                                      color=(1,1,1,0), width=3, parent=self.pulse_group)
            line.visible = False
            self._pulse_pool.append({
                "visual": line,
                "active": False,
                "start_time": 0.0,
                "origin": np.zeros(3, dtype=np.float32),
                "radius": 0.0,
                "color": (1.0, 1.0, 1.0, 1.0),
                "m1": 0.0, "n1": 0.0, "m2": 0.0, "n2": 0.0,
                "phi1": 0.0, "phi2": 0.0,
                "energy": 0.0,
                # orientation
                "axis": np.array([0.0,0.0,1.0], dtype=np.float32),
                "angle": 0.0,
            })

        # ---------------- Controls (right column) ---------------- #
        ctrl = QtWidgets.QGridLayout(); row=0

        # Tempo / groove
        ctrl.addWidget(QtWidgets.QLabel("BPM"), row,0); self.bpm = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.bpm.setRange(20,300); self.bpm.setValue(120)
        self.bpm.valueChanged.connect(lambda v: self.audio.set_bpm(float(v))); self.bpm_lbl = QtWidgets.QLabel("120"); self.bpm.valueChanged.connect(lambda v: self.bpm_lbl.setText(str(v)))
        ctrl.addWidget(self.bpm, row,1); ctrl.addWidget(self.bpm_lbl, row,2); row+=1

        ctrl.addWidget(QtWidgets.QLabel("Swing %"), row,0); self.swing = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.swing.setRange(0,50); self.swing.setValue(0)
        self.swing.valueChanged.connect(lambda v: self.audio.set_swing(v/100.0)); self.swing_lbl = QtWidgets.QLabel("0"); self.swing.valueChanged.connect(lambda v: self.swing_lbl.setText(str(v)))
        ctrl.addWidget(self.swing, row,1); ctrl.addWidget(self.swing_lbl, row,2); row+=1

        # Waves
        ctrl.addWidget(QtWidgets.QLabel("Bass Wave"), row,0); self.bwav = QtWidgets.QComboBox(); self.bwav.addItems(["saw","square","sine","triangle","pulse","supersaw"]); self.bwav.setCurrentText("saw")
        self.bwav.currentTextChanged.connect(lambda _: self.audio.set_voice("bass", shape=self.bwav.currentText()))
        ctrl.addWidget(self.bwav, row,1); row+=1

        ctrl.addWidget(QtWidgets.QLabel("Lead Wave"), row,0); self.lwav = QtWidgets.QComboBox(); self.lwav.addItems(["square","saw","sine","triangle","pulse","supersaw"]); self.lwav.setCurrentText("square")
        self.lwav.currentTextChanged.connect(lambda _: self.audio.set_voice("lead", shape=self.lwav.currentText()))
        ctrl.addWidget(self.lwav, row,1); row+=1

        # PWM / Detune / Cutoffs
        ctrl.addWidget(QtWidgets.QLabel("PWM (bass/lead)"), row,0); self.pwmb = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.pwml = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        for s in (self.pwmb, self.pwml): s.setRange(5,95); s.setValue(50); s.setTracking(False)
        self.pwmb.valueChanged.connect(lambda v: self.audio.set_voice("bass", pwm=v/100.0))
        self.pwml.valueChanged.connect(lambda v: self.audio.set_voice("lead", pwm=v/100.0))
        ctrl.addWidget(self.pwmb, row,1); ctrl.addWidget(self.pwml, row,2); row+=1

        ctrl.addWidget(QtWidgets.QLabel("Detune (cents) b/l"), row,0); self.detb = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.detl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        for s in (self.detb, self.detl): s.setRange(0,25); s.setValue(7); s.setTracking(False)
        self.detb.valueChanged.connect(lambda v: self.audio.set_voice("bass", detune=float(v)))
        self.detl.valueChanged.connect(lambda v: self.audio.set_voice("lead", detune=float(v)))
        ctrl.addWidget(self.detb, row,1); ctrl.addWidget(self.detl, row,2); row+=1

        ctrl.addWidget(QtWidgets.QLabel("Cutoff (Hz) b/l"), row,0); self.cutb = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.cutl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.cutb.setRange(50,2000); self.cutl.setRange(200,8000); self.cutb.setValue(220); self.cutl.setValue(1800)
        self.cutb.valueChanged.connect(lambda v: self.audio.set_voice("bass", cutoff=float(v)))
        self.cutl.valueChanged.connect(lambda v: self.audio.set_voice("lead", cutoff=float(v)))
        ctrl.addWidget(self.cutb, row,1); ctrl.addWidget(self.cutl, row,2); row+=1

        # Drive
        ctrl.addWidget(QtWidgets.QLabel("Drive (x0.1)"), row,0); self.drive = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.drive.setRange(5,30); self.drive.setValue(12)
        self.drive.valueChanged.connect(lambda v: self.audio.set_drive(v/10.0)); self.drive_lbl = QtWidgets.QLabel("1.2"); self.drive.valueChanged.connect(lambda v: self.drive_lbl.setText(f"{v/10:.1f}"))
        ctrl.addWidget(self.drive, row,1); ctrl.addWidget(self.drive_lbl, row,2); row+=1

        # Vibrato
        ctrl.addWidget(QtWidgets.QLabel("Vibrato (cents / Hz)"), row,0); self.vdep = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.vrate = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.vdep.setRange(0,50); self.vdep.setValue(4); self.vrate.setRange(1,120); self.vrate.setValue(55)
        self.vdep.valueChanged.connect(lambda _: self.audio.set_vibrato(float(self.vdep.value()), float(self.vrate.value()/10)))
        self.vrate.valueChanged.connect(lambda _: self.audio.set_vibrato(float(self.vdep.value()), float(self.vrate.value()/10)))
        ctrl.addWidget(self.vdep, row,1); ctrl.addWidget(self.vrate, row,2); row+=1

        # Delay
        ctrl.addWidget(QtWidgets.QLabel("Delay (ms / fb% / mix%)"), row,0); self.dms = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.dfb = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.dmx = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.dms.setRange(1,1500); self.dfb.setRange(0,95); self.dmx.setRange(0,100); self.dms.setValue(350); self.dfb.setValue(35); self.dmx.setValue(25)
        for s in (self.dms,self.dfb,self.dmx): s.setTracking(False)
        def upd_delay(_=None): self.audio.set_delay(float(self.dms.value()), float(self.dfb.value()/100), float(self.dmx.value()/100))
        self.dms.valueChanged.connect(upd_delay); self.dfb.valueChanged.connect(upd_delay); self.dmx.valueChanged.connect(upd_delay)
        ctrl.addWidget(self.dms, row,1); ctrl.addWidget(self.dfb, row,2); row+=1

        # Randomizer
        rand = QtWidgets.QPushButton("Randomize Patterns")
        rand.clicked.connect(self.audio.randomize_patterns)
        ctrl.addWidget(rand, row,0,1,3); row+=1

        # ============ Melodic Sphere Controls ============ #
        title_melodic = QtWidgets.QLabel("Sphere Melodic Influence"); title_melodic.setStyleSheet("color:#fcf; font-weight:bold;")
        ctrl.addWidget(title_melodic, row,0,1,3); row+=1

        # Overall influence
        ctrl.addWidget(QtWidgets.QLabel("Sphere→Melody"), row,0)
        self.sphere_influence = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sphere_influence.setRange(0, 100); self.sphere_influence.setValue(70)
        self.sphere_influence.valueChanged.connect(lambda v: setattr(self.melodic_engine, 'sphere_melody_influence', v/100.0))
        self.inf_lbl = QtWidgets.QLabel("70%"); self.sphere_influence.valueChanged.connect(lambda v: self.inf_lbl.setText(f"{v}%"))
        ctrl.addWidget(self.sphere_influence, row,1); ctrl.addWidget(self.inf_lbl, row,2); row+=1

        # Individual sphere weights
        ctrl.addWidget(QtWidgets.QLabel("S1 (Rhythm)"), row,0)
        self.s1_weight = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.s1_weight.setRange(0, 100); self.s1_weight.setValue(40)
        self.s1_weight.valueChanged.connect(lambda v: setattr(self.melodic_engine, 's1_melody_weight', v/100.0))
        ctrl.addWidget(self.s1_weight, row,1); row+=1

        ctrl.addWidget(QtWidgets.QLabel("S2 (Melody)"), row,0)
        self.s2_weight = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.s2_weight.setRange(0, 100); self.s2_weight.setValue(80)
        self.s2_weight.valueChanged.connect(lambda v: setattr(self.melodic_engine, 's2_melody_weight', v/100.0))
        ctrl.addWidget(self.s2_weight, row,1); row+=1

        ctrl.addWidget(QtWidgets.QLabel("S3 (Harmony)"), row,0)
        self.s3_weight = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.s3_weight.setRange(0, 100); self.s3_weight.setValue(60)
        self.s3_weight.valueChanged.connect(lambda v: setattr(self.melodic_engine, 's3_melody_weight', v/100.0))
        ctrl.addWidget(self.s3_weight, row,1); row+=1

        # Tension tracking
        self.tension_check = QtWidgets.QCheckBox("Harmonic Tension Tracking")
        self.tension_check.setChecked(True)
        self.tension_check.stateChanged.connect(lambda s: setattr(self.melodic_engine, 'melodic_tension_tracking', s==2))
        ctrl.addWidget(self.tension_check, row,0,1,3); row+=1

        # Counterpoint
        title = QtWidgets.QLabel("Counterpoint"); title.setStyleSheet("color:#8cf; font-weight:bold;")
        ctrl.addWidget(title, row,0,1,3); row+=1

        self.cp_enable = QtWidgets.QCheckBox("Enable")
        self.cp_enable.stateChanged.connect(lambda s: self.audio.set_counterpoint(enabled=(s==2)))
        ctrl.addWidget(self.cp_enable, row,0,1,3); row+=1

        ctrl.addWidget(QtWidgets.QLabel("CP Scale"), row,0)
        self.cp_scale = QtWidgets.QComboBox()
        self.cp_scale.addItems(["S2-derived","Hirajoshi","Pelog-ish","Messiaen-3","Bhairav-ish"])
        self.cp_scale.currentTextChanged.connect(lambda name: self.audio.set_counterpoint(scale_name=name))
        ctrl.addWidget(self.cp_scale, row,1); row+=1

        ctrl.addWidget(QtWidgets.QLabel("CP Lag (steps)"), row,0)
        self.cp_lag = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.cp_lag.setRange(0, 8); self.cp_lag.setValue(2)
        self.cp_lag.valueChanged.connect(lambda v: self.audio.set_counterpoint(lag_steps=int(v)))
        ctrl.addWidget(self.cp_lag, row,1); row+=1

        self.cp_invert = QtWidgets.QCheckBox("Invert (contrary motion)")
        self.cp_invert.setChecked(True)
        self.cp_invert.stateChanged.connect(lambda s: self.audio.set_counterpoint(invert=(s==2)))
        ctrl.addWidget(self.cp_invert, row,0,1,3); row+=1

        ctrl.addWidget(QtWidgets.QLabel("CP Level"), row,0)
        self.cp_lvl = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.cp_lvl.setRange(0, 100); self.cp_lvl.setValue(18)
        self.cp_lvl.valueChanged.connect(lambda v: self.audio.set_counterpoint(level=v/100.0))
        ctrl.addWidget(self.cp_lvl, row,1); row+=1

        ctrl.addWidget(QtWidgets.QLabel("CP Cutoff"), row,0)
        self.cp_cut = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.cp_cut.setRange(200, 12000); self.cp_cut.setValue(2400)
        self.cp_cut.valueChanged.connect(lambda v: self.audio.set_cp_voice(cutoff=float(v)))
        ctrl.addWidget(self.cp_cut, row,1); row+=1

        ctrl.addWidget(QtWidgets.QLabel("CP Wave"), row,0)
        self.cp_wave = QtWidgets.QComboBox(); self.cp_wave.addItems(["sine","triangle","square","saw","pulse","supersaw"])
        self.cp_wave.currentTextChanged.connect(lambda w: self.audio.set_cp_voice(shape=w))
        ctrl.addWidget(self.cp_wave, row,1); row+=1

        # Cymatic Pulse controls
        title2 = QtWidgets.QLabel("Cymatic Pulses"); title2.setStyleSheet("color:#cfa; font-weight:bold;")
        ctrl.addWidget(title2, row,0,1,3); row+=1

        self.cym_enable = QtWidgets.QCheckBox("Enable")
        self.cym_enable.setChecked(True)
        self.cym_enable.stateChanged.connect(lambda s: setattr(self, "cym_pulses_enabled", s==2))
        ctrl.addWidget(self.cym_enable, row,0,1,3); row+=1

        ctrl.addWidget(QtWidgets.QLabel("Max Rings"), row,0)
        self.ring_count = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.ring_count.setRange(4, 48); self.ring_count.setValue(self.max_pulse_rings)
        def upd_ring_count(v):
            v = int(v)
            if v == self.max_pulse_rings: return
            if v < len(self._pulse_pool):
                for slot in self._pulse_pool[v:]:
                    slot["active"] = False
                    slot["visual"].visible = False
                    slot["visual"].set_data(np.zeros((0,3), dtype=np.float32))
                self._pulse_pool = self._pulse_pool[:v]
            else:
                for _ in range(v - len(self._pulse_pool)):
                    line = scene.visuals.Line(np.zeros((0,3), dtype=np.float32),
                                              color=(1,1,1,0), width=3, parent=self.pulse_group)
                    line.visible = False
                    self._pulse_pool.append({
                        "visual": line, "active": False, "start_time": 0.0,
                        "origin": np.zeros(3, dtype=np.float32), "radius": 0.0,
                        "color": (1.0,1.0,1.0,1.0),
                        "m1":0.0,"n1":0.0,"m2":0.0,"n2":0.0,"phi1":0.0,"phi2":0.0,"energy":0.0,
                        "axis": np.array([0.0,0.0,1.0], dtype=np.float32), "angle": 0.0
                    })
            self.max_pulse_rings = v
        self.ring_count.valueChanged.connect(upd_ring_count)
        ctrl.addWidget(self.ring_count, row,1); row+=1

        ctrl.addWidget(QtWidgets.QLabel("Ring Lifetime (s)"), row,0)
        self.ring_life = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.ring_life.setRange(10, 80); self.ring_life.setValue(int(self.pulse_lifetime*10))
        self.ring_life.valueChanged.connect(lambda v: setattr(self, "pulse_lifetime", float(v)/10.0))
        ctrl.addWidget(self.ring_life, row,1); row+=1

        right.addLayout(ctrl)

        # Stats
        self.stats = QtWidgets.QPlainTextEdit(); self.stats.setReadOnly(True); self.stats.setMinimumWidth(520); self.stats.setMinimumHeight(250)
        right.addWidget(self.stats, 1)

        # Origin marker + axes
        self.origin_marker = scene.visuals.Markers(parent=self.view3d.scene)
        self.origin_marker.set_data(pos=np.array([[0,0,0]]), face_color="white", edge_color="yellow", edge_width=2, size=18, symbol="disc")
        self.x_axis = scene.visuals.XYZAxis(parent=self.view3d.scene, width=2)

        # Timers
        self._vis_frame = 0; self._sim_frame = 0
        self.timer = QtCore.QTimer(); self.timer.setInterval(40); self.timer.timeout.connect(self._tick); self.timer.start()

        # ---- Hook audio events AFTER fields are ready (avoids race) ---- #
        self.audio.on_step_event = self._on_step

        print("[engine] Berlin × Tri-Harmonic running (Enhanced Melodic + Counterpoint + 3D cymatic pulses + controls)")

    # Geometry
    def _create_torus_wireframe(self):
        R = self.core.major_radius; r = self.core.minor_radius
        u = np.linspace(0, 2*np.pi, 30); v = np.linspace(0, 2*np.pi, 20)
        for i in range(len(u)):
            x = (R + r*np.cos(v))*np.cos(u[i]); y = (R + r*np.cos(v))*np.sin(u[i]); z = r*np.sin(v)
            scene.visuals.Line(np.column_stack([x,y,z]), color=(0.5,0.5,0.5,0.2), parent=self.view3d.scene, width=0.5)
        for j in range(0, len(v), 2):
            x = (R + r*np.cos(v[j]))*np.cos(u); y = (R + r*np.cos(v[j]))*np.sin(u); z = r*np.sin(v[j])*np.ones_like(u)
            scene.visuals.Line(np.column_stack([x,y,z]), color=(0.5,0.5,0.5,0.2), parent=self.view3d.scene, width=0.5)

    def _create_sphere_wireframes(self):
        th = np.linspace(0, 2*np.pi, 20); r = self.core.sphere_s1.radius
        scene.visuals.Line(np.column_stack([r*np.cos(th), r*np.sin(th), 0*th]), color="orange", parent=self.view3d.scene, width=2)
        if self.core.sphere_s2: scene.visuals.Line(np.column_stack([0*th, r*np.cos(th), r*np.sin(th)]), color="cyan", parent=self.view3d.scene, width=2)
        if self.core.sphere_s3: scene.visuals.Line(np.column_stack([r*np.cos(th), 0*th, r*np.sin(th)]), color="lime", parent=self.view3d.scene, width=2)

    # --- Math helpers for 3D ring orientation --------------------------------#
    @staticmethod
    def _normalize(v):
        n = np.linalg.norm(v)
        return v if n < 1e-9 else v / n

    @staticmethod
    def _rot_matrix(axis, angle):
        axis = BerlinTriVis._normalize(axis.astype(np.float32))
        x, y, z = axis
        c = np.cos(angle); s = np.sin(angle); C = 1.0 - c
        return np.array([
            [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
            [y*x*C + z*s,   c + y*y*C,   y*z*C - x*s],
            [z*x*C - y*s,   z*y*C + x*s, c + z*z*C  ]
        ], dtype=np.float32)

    # Event from audio step
    def _on_step(self, kind: str, freq: float, vel: float, step_idx:int):
        if not getattr(self, "cym_pulses_enabled", True):
            return
        u_phase = (step_idx / 16.0) * 2*np.pi
        energy = float(np.clip(vel, 0.05, 1.0))
        sym = f"{kind.UPPER()}-{int(freq)}Hz" if hasattr(kind, "UPPER") else f"{str(kind).upper()}-{int(freq)}Hz"
        dp = DataPoint(value=freq, timestamp=time.time(), symbol=sym, phase=u_phase,
                       spectrum=None, energy=energy, duration=0.25, metadata={})
        self.core.process(dp)
        origin = self.core._torus_pos(u_phase, energy)
        self._make_pulse(freq, energy, origin, u_phase)

    # Pulse visuals (pooled — cymatic-deformed rings, **tilted by pitch**, **coloured by note**)
    def _make_pulse(self, freq, energy, origin, u_phase):
        try:
            slot = self._alloc_pulse()
            th = self._pulse_theta
            r0 = 2.0

            # ---- Colour by NOTE (pitch class relative to root) ----
            pc = int(np.round(hz_to_semitones(freq, self.audio.root))) % 12
            hue = (pc / 12.0)  # wrap around colour wheel per semitone
            sat = 0.85
            val = 0.6 + 0.4 * float(np.clip(energy, 0.0, 1.0))
            r, g, b = hsv_to_rgb(hue, sat, val)
            col = (r, g, b, 1.0)

            # cymatic snapshot from current bass/lead freqs
            bfreq = float(self.audio.b_freq); lfreq = float(self.audio.l_freq)
            m1, n1 = self._freq_to_modes(bfreq)
            m2, n2 = self._freq_to_modes(lfreq)
            t = self.audio.time
            phi1 = 0.35 * t
            phi2 = 0.22 * t

            # deformation in ring's local plane
            amp = 0.6 + 0.8*energy
            deform = (0.55*np.sin(m1*th + phi1) + 0.45*np.cos(m2*th + phi2))
            radius = r0 + amp*0.6*deform

            # --- 3D orientation: rotate ring plane around torus tangent by pitch angle
            tangent = np.array([-np.sin(u_phase), np.cos(u_phase), 0.0], dtype=np.float32)
            tilt = np.deg2rad(55.0) * np.clip(hz_to_semitones(freq, self.audio.root)/24.0, -1.0, 1.0)
            R = self._rot_matrix(tangent, float(tilt))

            # build local ring in XY plane around origin, then rotate by R
            local = np.stack([radius*np.cos(th), radius*np.sin(th), np.zeros_like(th)], axis=1).astype(np.float32)
            rotated = local @ R.T
            pos = rotated + np.asarray(origin, dtype=np.float32)

            v = slot["visual"]
            v.set_data(pos, color=col)
            v.visible = True

            slot.update({
                "active": True,
                "start_time": time.time(),
                "origin": np.array(origin, dtype=np.float32),
                "radius": r0,
                "color": col,
                "m1": m1, "n1": n1, "m2": m2, "n2": n2,
                "phi1": phi1, "phi2": phi2,
                "energy": energy,
                "axis": tangent,
                "angle": float(tilt),
            })
        except Exception as e:
            print(f"[pulse create] {e}")

    # Pool mgmt
    def _alloc_pulse(self):
        for r in self._pulse_pool:
            if not r["active"]:
                return r
        oldest = min(self._pulse_pool, key=lambda r: r["start_time"])
        v = oldest["visual"]
        v.visible = False
        v.set_data(np.zeros((0,3), dtype=np.float32))
        oldest["active"] = False
        return oldest

    @staticmethod
    def _freq_to_modes(f, base=55.0):
        ratio = max(0.1, f/base)
        m = 2.0 + 6.0*np.log2(ratio)
        n = 2.0 + 5.0*np.sqrt(max(0.0, ratio - 0.5))
        return m, n

    def _update_pulse_rings(self):
        if not self._pulse_pool:
            return
        now = time.time()
        th = self._pulse_theta
        for slot in self._pulse_pool:
            if not slot["active"]:
                continue
            v = slot["visual"]
            try:
                age = now - slot["start_time"]
                max_age = float(self.pulse_lifetime)
                if age >= max_age or not np.isfinite(age):
                    slot["active"] = False
                    v.visible = False
                    v.set_data(np.zeros((0,3), dtype=np.float32))
                    continue

                prog = age / max_age
                base_r = slot["radius"] + prog * 20.0
                alpha = max(0.0, (1.0 - prog) * 0.9)

                # animate cymatic phases while expanding
                phi1 = slot["phi1"] + 0.6 * age
                phi2 = slot["phi2"] + 0.45 * age
                amp = 0.6 + 0.8*slot["energy"]
                deform = (0.55*np.sin(slot["m1"]*th + phi1) + 0.45*np.cos(slot["m2"]*th + phi2))
                radius = base_r + amp*0.6*deform

                # rotate into 3D using stored axis/angle
                R = self._rot_matrix(slot["axis"], slot["angle"])
                local = np.stack([radius*np.cos(th), radius*np.sin(th), np.zeros_like(th)], axis=1).astype(np.float32)
                rotated = local @ R.T
                pos = rotated + slot["origin"]

                base = slot["color"]
                v.set_data(pos, color=(base[0], base[1], base[2], alpha))
            except Exception:
                slot["active"] = False
                v.visible = False
                v.set_data(np.zeros((0,3), dtype=np.float32))

    def _update_particles(self):
        if self.core.torus_buffer:
            n = min(40, len(self.core.torus_buffer)); samples = list(self.core.torus_buffer)[-n:]
            pos=[]
            for d in samples:
                u = d.phase + self._vis_frame*0.02; v = d.energy*2*np.pi
                R, rr = self.core.major_radius, self.core.minor_radius*0.9
                x=(R+rr*np.cos(v))*np.cos(u); y=(R+rr*np.cos(v))*np.sin(u); z=rr*np.sin(v)
                pos.append([x,y,z])
            self.torus_particles.set_data(np.array(pos), face_color="yellow", edge_color="white", edge_width=0.5, size=10)
        else:
            self.torus_particles.set_data(pos=np.zeros((0,3)))

        def upd_sphere(s, particles, col):
            pts=[]
            for d in s.memory_ring:
                if "sphere_position" in (d.metadata or {}): pts.append(d.metadata["sphere_position"])
            if pts: particles.set_data(np.array(pts), face_color=col, edge_color="white", edge_width=0.5, size=14, symbol="star")
            else: particles.set_data(pos=np.zeros((0,3)))
        upd_sphere(self.core.sphere_s1, self.s1_particles, "orange")
        if self.core.sphere_s2: upd_sphere(self.core.sphere_s2, self.s2_particles, "cyan")
        if self.core.sphere_s3: upd_sphere(self.core.sphere_s3, self.s3_particles, "lime")

    def _update_stats(self):
        st = self.core.metrics; total = st["total_processed"]; tor = st["torus_direct"]
        eff = (100 - (tor/total*100)) if total>0 else 0
        
        # Get melodic data
        melodic_data = self.melodic_engine.extract_sphere_melodic_data()
        tension_curve = self.melodic_engine.tension_curve
        
        s = "BERLIN × TRI-HARMONIC — ENHANCED MELODIC STATS\n\n"
        s += f"Processed: {total}\n"
        s += f"S1 entries: {st['s1_entries']}  | S2: {st.get('s2_entries',0)}  | S3: {st.get('s3_entries',0)}\n"
        s += f"Torus direct: {tor}  | Routing efficiency: {eff:.1f}%\n"
        s += f"Freqs: Bass {self.audio.b_freq:.1f} Hz  | Lead {self.audio.l_freq:.1f} Hz\n"
        s += f"Delay: {self.audio.delay_ms:.0f} ms  fb {self.audio.delay_fb*100:.0f}%  mix {self.audio.delay_mix*100:.0f}%\n"
        s += f"CP: {'ON' if self.audio.cp_enabled else 'off'}  | Scale: {self.audio.cp_scale_name}  | Lag:{self.audio.cp_lag_steps} inv:{self.audio.cp_invert}\n"
        s += f"Pulses: {'ON' if self.cym_pulses_enabled else 'off'}  | Pool:{len(self._pulse_pool)}  | Life:{self.pulse_lifetime:.1f}s\n\n"
        
        # Melodic sphere stats
        s += "MELODIC SPHERE DATA:\n"
        s += f"Sphere Influence: {self.melodic_engine.sphere_melody_influence*100:.0f}%\n"
        s += f"Temporal Energy: {melodic_data['temporal_energy']:.2f}\n"
        s += f"Semantic Drift: {melodic_data['semantic_drift']:.2f}\n"
        s += f"Harmonic Tension: {melodic_data['harmonic_tension']:.2f}\n"
        s += f"Tension Curve: {tension_curve:.2f}\n"
        
        if melodic_data['phase_relationships']:
            pr = melodic_data['phase_relationships']
            s += f"Phase Regularity: {pr['regularity']:.2f}\n"
            s += f"Phase Acceleration: {pr['acceleration']:.2f}\n"
        
        s += f"Symbol Density: {melodic_data['symbol_density']:.2f}\n"
        s += f"Phrase Memory: {len(self.melodic_engine.phrase_memory)}/32\n"
        
        self.stats.setPlainText(s)

    def _tick(self):
        self._vis_frame += 1; self._sim_frame += 1
        self._update_pulse_rings(); self._update_particles(); self._update_stats()

    def run(self):
        self.canvas.show(); self.win.resize(1720, 1050); self.win.show()
        try:
            self.app.exec_()
        finally:
            self.audio.stop()

# -----------------------------------------------------------------------------#
# Main
# -----------------------------------------------------------------------------#
def main():
    print("\n" + "="*96)
    print("  BERLIN × TRI-HARMONIC — Enhanced Melodic • 16-Step Sequencer • Spheres • 3D Cymatic Pulses • Delay • Counterpoint")
    print("="*96)
    ui = BerlinTriVis()
    ui.run()

if __name__ == "__main__":
    main()