import unittest
import numpy as np
from unittest.mock import patch, MagicMock

from src.language_classifier import LanguageClassifier
from src.config.settings import LABEL_TO_NAME, NUM_CLASSES


class TestLanguageClassifier(unittest.TestCase):

    def test_init(self):
        clf = LanguageClassifier()
        self.assertIsNone(clf.manifest)
        self.assertIsNotNone(clf.classifier)

    def test_predict_returns_expected_keys(self):
        """Verify predict() returns dict with required keys."""
        clf = LanguageClassifier()

        # Mock the internal dependencies
        clf.extractor = MagicMock()
        clf.extractor.extract_single.return_value = np.zeros(1024)

        clf.classifier.best_model = MagicMock()
        clf.classifier.best_model.predict.return_value = np.array([0])
        clf.classifier.best_model.predict_proba.return_value = np.ones((1, NUM_CLASSES)) / NUM_CLASSES
        clf.classifier.scaler = MagicMock()
        clf.classifier.scaler.transform.return_value = np.zeros((1, 1024))

        with patch('src.language_classifier.preprocess_audio', return_value=np.zeros(48000)):
            result = clf.predict('fake_audio.wav')

        self.assertIn('language', result)
        self.assertIn('language_code', result)
        self.assertIn('confidence', result)
        self.assertIn(result['language'], LABEL_TO_NAME.values())

    def test_label_mappings(self):
        """Verify label mappings are consistent."""
        self.assertEqual(len(LABEL_TO_NAME), NUM_CLASSES)
        for i in range(NUM_CLASSES):
            self.assertIn(i, LABEL_TO_NAME)


if __name__ == '__main__':
    unittest.main()