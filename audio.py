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
