import unittest
import numpy as np

from src.preprocessing import (
    normalize_audio, pad_or_trim, is_valid_audio,
)
from src.config.settings import SAMPLE_RATE, TARGET_SAMPLES


class TestPreprocessing(unittest.TestCase):

    def test_normalize_audio(self):
        y = np.array([0.5, -0.5, 0.2, -0.2], dtype=np.float32)
        normed = normalize_audio(y)
        self.assertAlmostEqual(float(np.max(np.abs(normed))), 1.0)

    def test_normalize_silent(self):
        y = np.zeros(100, dtype=np.float32)
        normed = normalize_audio(y)
        self.assertTrue(np.allclose(normed, 0.0))

    def test_pad_or_trim_pad(self):
        y = np.ones(100, dtype=np.float32)
        result = pad_or_trim(y, target_len=200)
        self.assertEqual(len(result), 200)
        self.assertTrue(np.allclose(result[100:], 0.0))

    def test_pad_or_trim_trim(self):
        y = np.ones(200, dtype=np.float32)
        result = pad_or_trim(y, target_len=100)
        self.assertEqual(len(result), 100)

    def test_pad_or_trim_exact(self):
        y = np.ones(TARGET_SAMPLES, dtype=np.float32)
        result = pad_or_trim(y)
        self.assertEqual(len(result), TARGET_SAMPLES)

    def test_is_valid_audio_short(self):
        y = np.ones(100, dtype=np.float32)  # < 0.5s at 16kHz
        self.assertFalse(is_valid_audio(y))

    def test_is_valid_audio_silent(self):
        y = np.zeros(SAMPLE_RATE, dtype=np.float32)
        self.assertFalse(is_valid_audio(y))

    def test_is_valid_audio_ok(self):
        y = np.random.randn(SAMPLE_RATE).astype(np.float32)
        self.assertTrue(is_valid_audio(y))


if __name__ == '__main__':
    unittest.main()