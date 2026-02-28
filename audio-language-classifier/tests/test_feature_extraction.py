import unittest
import numpy as np
from unittest.mock import patch, MagicMock

from src.config.settings import EMBEDDING_DIM, SAMPLE_RATE, TARGET_SAMPLES


class TestFeatureExtraction(unittest.TestCase):

    def test_extract_single_shape(self):
        """Verify extract_single returns correct shape (mocked model)."""
        from src.feature_extraction import Wav2VecExtractor

        with patch.object(Wav2VecExtractor, '__init__', lambda self, **kw: None):
            ext = Wav2VecExtractor.__new__(Wav2VecExtractor)
            ext.device = 'cpu'
            ext.processor = MagicMock()
            ext.model = MagicMock()

            # Mock model output
            mock_hidden = MagicMock()
            mock_hidden.mean.return_value = MagicMock()
            mock_hidden.mean.return_value.cpu.return_value.numpy.return_value = np.zeros((1, EMBEDDING_DIM))

            ext.model.return_value = MagicMock(last_hidden_state=mock_hidden)
            ext.processor.return_value = MagicMock(
                input_values=MagicMock(to=MagicMock(return_value=MagicMock()))
            )

            waveform = np.random.randn(TARGET_SAMPLES).astype(np.float32)
            result = ext.extract_single(waveform)

            self.assertEqual(result.shape, (EMBEDDING_DIM,))

    def test_embedding_dim_constant(self):
        self.assertEqual(EMBEDDING_DIM, 1024)


if __name__ == '__main__':
    unittest.main()