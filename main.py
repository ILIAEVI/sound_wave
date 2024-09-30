import numpy as np
from scipy.io import wavfile
import os
import re


class SoundWaveFactory:

    NOTES = {
        '0': 0, 'e0': 20.60172, 'f0': 21.82676, 'f#0': 23.12465, 'g0': 24.49971, 'g#0': 25.95654,
        'a0': 27.50000, 'a#0': 29.13524, 'b0': 30.86771, 'c0': 32.70320, 'c#0': 34.64783,
        'd0': 36.70810, 'd#0': 38.89087, 'e1': 41.20344, 'f1': 43.65353, 'f#1': 46.24930,
        'g1': 48.99943, 'g#1': 51.91309, 'a1': 55.00000, 'a#1': 58.27047, 'b1': 61.73541,
        'c1': 65.40639, 'c#1': 69.29566, 'd1': 73.41619, 'd#1': 77.78175, 'e2': 82.40689,
        'f2': 87.30706, 'f#2': 92.49861, 'g2': 97.99886, 'g#2': 103.8262, 'a2': 110.0000,
        'a#2': 116.5409, 'b2': 123.4708, 'c2': 130.8128, 'c#2': 138.5913, 'd2': 146.8324,
        'd#2': 155.5635, 'e3': 164.8138, 'f3': 174.6141, 'f#3': 184.9972, 'g3': 195.9977,
        'g#3': 207.6523, 'a3': 220.0000, 'a#3': 233.0819, 'b3': 246.9417, 'c3': 261.6256,
        'c#3': 277.1826, 'd3': 293.6648, 'd#3': 311.1270, 'e4': 329.6276, 'f4': 349.2282,
        'f#4': 369.9944, 'g4': 391.9954, 'g#4': 415.3047, 'a4': 440.0000, 'a#4': 466.1638,
        'b4': 493.8833, 'c4': 523.2511, 'c#4': 554.3653, 'd4': 587.3295, 'd#4': 622.2540,
        'e5': 659.2551, 'f5': 698.4565, 'f#5': 739.9888, 'g5': 783.9909, 'g#5': 830.6094,
        'a5': 880.0000, 'a#5': 932.3275, 'b5': 987.7666, 'c5': 1046.502, 'c#5': 1108.731,
        'd5': 1174.659, 'd#5': 1244.508, 'e6': 1318.510, 'f6': 1396.913, 'f#6': 1479.978,
        'g6': 1567.982, 'g#6': 1661.219, 'a6': 1760.000, 'a#6': 1864.655, 'b6': 1975.533,
        'c6': 2093.005, 'c#6': 2217.461, 'd6': 2349.318, 'd#6': 2489.016, 'e7': 2637.020,
        'f7': 2793.826, 'f#7': 2959.955, 'g7': 3135.963, 'g#7': 3322.438, 'a7': 3520.000,
        'a#7': 3729.310, 'b7': 3951.066, 'c7': 4186.009, 'c#7': 4434.922, 'd7': 4698.636,
        'd#7': 4978.032,
    }

    def __init__(self, sampling_rate=44100, duration_seconds=5, max_amplitude= 2 ** 13):
        self.sampling_rate = sampling_rate
        self.duration_seconds = duration_seconds
        self.sound_array_len = self.sampling_rate * self.duration_seconds
        self.max_amplitude = max_amplitude
        self.common_timeline = np.linspace(0, self.duration_seconds, self.sound_array_len)

    def _get_normed_sin(self, frequency):
        """Generate normalized sine wave based on timeline and frequency."""
        return self.max_amplitude * np.sin(2 * np.pi * frequency * self.common_timeline)

    def create_note(self, note="a4", name=None):
        """Create a sine wave for a given note and save it as a WAV file."""
        if note not in self.NOTES:
            raise ValueError(f"Note {note} is not valid. Please choose from: {list(self.NOTES.keys())}")

        sound_wave = self._get_normed_sin(self.NOTES[note]).astype(np.int16)

        file_name = f"{note}_sin.wav".replace("#", "s") if name is None else f"{name}.wav"
        wavfile.write(file_name, self.sampling_rate, sound_wave)
        return sound_wave

    @staticmethod
    def read_wave_from_txt(txt_file):
        """Read a wave from a .txt file."""
        if not os.path.exists(txt_file):
            raise FileNotFoundError(f"File {txt_file} not found.")

        wave_data = np.loadtxt(txt_file, dtype=np.int16)
        return wave_data

    @staticmethod
    def print_wave_details(wave_data):
        """Print details about the wave, including mean, std deviation, and length."""
        if not isinstance(wave_data, np.ndarray):
            raise ValueError("Input must be a numpy array.")

        print(f"Wave Length: {len(wave_data)} samples")
        print(f"Mean Amplitude: {np.mean(wave_data)}")
        print(f"Standard Deviation: {np.std(wave_data)}")

    def normalize_sound_waves(self, *waves):
        """Normalize multiple sound waves to the shortest length and scale amplitude."""
        min_length = min(len(wave) for wave in waves)
        normalized_waves = [wave[:min_length] for wave in waves]

        max_amplitude = max(np.max(np.abs(wave)) for wave in normalized_waves)
        normalized_waves = [wave * (self.max_amplitude / max_amplitude) for wave in normalized_waves]

        return np.array(normalized_waves)

    def save_wave(self, wave_data, file_name='output', file_type='txt'):
        """Save the wave into a .txt file or WAV file."""
        if file_type.lower() == 'wav':
            wavfile.write(f"{file_name}.wav", self.sampling_rate, wave_data)
        elif file_type.lower() == 'txt':
            np.savetxt(f"{file_name}", wave_data, fmt='%d')
        else:
            raise ValueError("Unsupported file type. Use 'txt' or 'WAV'.")

    def convert_wave_type(self, wave_data, wave_type='sine'):
        """Convert sine waves to triangular or square waves."""
        if wave_type not in ['sine', 'triangle', 'square']:
            raise ValueError("wave_type must be 'sine', 'triangle', or 'square'.")

        if wave_type == 'triangle':
            return np.int16(self.max_amplitude * (2 * np.abs(2 * (wave_data / self.max_amplitude) - 1) - 1))
        elif wave_type == 'square':
            return np.int16(self.max_amplitude * np.sign(wave_data))

        return wave_data

    def combine_waves(self, *waves):
        """Combine multiple sound waves into one."""
        max_length = max(len(wave) for wave in waves)
        combined_wave = np.zeros(max_length)

        for wave in waves:
            combined_wave[:len(wave)] += wave

        combined_wave = np.int16(combined_wave / np.max(np.abs(combined_wave)) * self.max_amplitude)

        return combined_wave

    def generate_melody_from_text(self, text):
        """Generate a melody from a text string of notes."""
        note_pattern = r'(\w#?\d)(\s*\d*\.?\d*s)?'
        matches = re.findall(note_pattern, text)
        waves = []

        for match in matches:
            note, duration = match
            duration = duration.strip() if duration else '1s'

            if note in self.NOTES:
                frequency = self.NOTES[note]
                duration_seconds = float(duration[:-1])

                sound_wave = self._get_normed_sin(frequency)[:int(self.sampling_rate * duration_seconds)].astype(
                    np.int16)
                waves.append(sound_wave)
            elif note.startswith('(') and note.endswith(')'):
                nested_notes = note[1:-1].strip().split()
                nested_wave = self.combine_waves(*[self.create_note(n) for n in nested_notes])
                waves.append(nested_wave)

        combined_wave = self.combine_waves(*waves)
        return combined_wave


if __name__ == "__main__":
    factory = SoundWaveFactory()
    main_note = 'b2'
    filename = f'{main_note}_sin.txt'

    a4_wave = factory.create_note(note=main_note)

    factory.save_wave(a4_wave, file_name=filename)

    factory.print_wave_details(a4_wave)

    loaded_wave = factory.read_wave_from_txt(filename)

    normed_waves = factory.normalize_sound_waves(a4_wave, loaded_wave)

    print('Normalized Waves:')

    factory.print_wave_details(normed_waves)

    melody_text = "g4 0.2s b4 0.2s (g3 d5 g5) 0.5s"
    melody_wave = factory.generate_melody_from_text(melody_text)

    factory.save_wave(melody_wave, file_name='melody', file_type='wav')
    print("Melody Waves:")
    factory.print_wave_details(melody_wave)