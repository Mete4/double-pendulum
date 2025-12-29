import numpy as np
from scipy.io import wavfile

class AudioEngine:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate

    def process_data(self, th1, th2, target_duration=20.0):
        """
        Interprets raw pendulum angles directly as sound waves.
        One channel for the top bob (Left), one for the bottom (Right).
        """
        num_output_samples = int(target_duration * self.sample_rate)
        
        # Take the maximum absolute value from both signals to normalize correctly
        max_val = max(np.max(np.abs(th1)), np.max(np.abs(th2)))
        if max_val < 1e-6: max_val = 1.0
        
        s1 = th1 / max_val
        s2 = th2 / max_val
        
        # Resample to 44.1kHz
        from scipy.interpolate import interp1d
        x_old = np.linspace(0, 1, len(th1))
        x_new = np.linspace(0, 1, num_output_samples)
        
        # Fast linear interpolation for audio
        sig_l = interp1d(x_old, s1, kind='linear')(x_new)
        sig_r = interp1d(x_old, s2, kind='linear')(x_new)
        
        # Combine into stereo
        audio_data = np.vstack((sig_l, sig_r)).T
        
        audio_data = np.clip(audio_data, -1, 1)
        return (audio_data * 32767).astype(np.int16)

    def save(self, data, path):
        wavfile.write(path, self.sample_rate, data)

    def synthesize_spectral(self, th1, th2, output_duration, input_dt, num_harmonics=16):
        """
        Analyzes the trajectory to find dominant frequencies and synthesizes 
        a clean, infinite loop based on that 'musical DNA'.
        """
        from scipy.fft import rfft, rfftfreq
        
        # Analyze the trajectory
        N = len(th1)
        
        window = np.hanning(N)
        
        def get_components(signal):
            # FFT
            yf = rfft(signal * window)
            xf = rfftfreq(N, d=input_dt)
            
            # Get Magnitude and Phase
            amplitudes = np.abs(yf)
            phases = np.angle(yf)
            
            # Find Peaks 
            if len(amplitudes) > 2:
                # Vectorized peak finding
                center = amplitudes[1:-1]
                left = amplitudes[:-2]
                right = amplitudes[2:]
                
                is_peak = (center > left) & (center > right)
                
                peak_indices = np.where(is_peak)[0] + 1
                
                sorted_peaks = sorted(peak_indices, key=lambda i: amplitudes[i], reverse=True)
                
                indices = sorted_peaks
            else:
                indices = []
            
            components = []
            count = 0
            for idx in indices:
                if idx == 0: continue # Skip DC
                
                freq = xf[idx]
                amp = amplitudes[idx]
                phase = phases[idx]
                
                components.append((freq, amp, phase))
                count += 1
                if count >= num_harmonics:
                    break
            return components

        comps_l = get_components(th1)
        comps_r = get_components(th2)
        
        # 2. Synthesize New Signal
        num_samples = int(output_duration * self.sample_rate)
        t = np.linspace(0, output_duration, num_samples, endpoint=False)
        
        # PITCH FACTOR
        PITCH_MULTIPLIER = 750.0 
        
        def synthesize_channel(components):
            signal = np.zeros(num_samples)
            total_amp = 0.0
            for freq, amp, phase in components:
                audible_freq = freq * PITCH_MULTIPLIER
                
                if audible_freq > self.sample_rate / 2: continue
                
                signal += amp * np.sin(2 * np.pi * audible_freq * t + phase)
                total_amp += amp
            
            if total_amp > 1e-6:
                signal /= total_amp
            return signal

        sig_l = synthesize_channel(comps_l)
        sig_r = synthesize_channel(comps_r)
        
        # Combine
        audio_data = np.vstack((sig_l, sig_r)).T
        return (np.clip(audio_data, -1, 1) * 32767).astype(np.int16)
