"""Public API guards: never silently change validation cohorts/checkpoint family."""
import tempfile
import unittest
from pathlib import Path

from waterec_ocr.waterec_compact_models import load_compact_checkpoint
from waterec_ocr.waterec_compact_workflow import train_compact


class CompactGuardTests(unittest.TestCase):
    def config(self, root, **changes):
        config = {'dataset_root': str(root / 'unused_dataset'),
                  'output_dir': str(root / 'not_created'), 'device': 'cpu',
                  'validation_fraction': 0, 'teacher_mode': 'provided'}
        config.update(changes)
        return config

    def test_explicit_missing_validation_directory_never_becomes_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.config(root, validation_folder=str(root / 'missing_validation'))
            with self.assertRaisesRegex(FileNotFoundError, 'validation_folder'):
                train_compact(config)
            self.assertFalse(Path(config['output_dir']).exists())

    def test_invalid_validation_fractions_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for fraction in (-.1, 1., 1.1, float('nan'), float('inf'), float('-inf')):
                with self.subTest(fraction=fraction):
                    config = self.config(root, validation_fraction=fraction)
                    with self.assertRaisesRegex(ValueError, 'validation_fraction'):
                        train_compact(config)
                    self.assertFalse(Path(config['output_dir']).exists())

    def test_provided_teacher_requires_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(Path(directory))
            with self.assertRaisesRegex(ValueError, 'teacher_checkpoint'):
                train_compact(config)
            self.assertFalse(Path(config['output_dir']).exists())

    def test_standard_or_other_family_checkpoint_is_not_guessed_as_compact(self):
        for payload in ({'config': {'task': 'price'}, 'state_dict': {}},
                        {'compact_architecture': {'family': 'waterec'}, 'state_dict': {}},
                        {'compact_architecture': {'family': 'waterec_compact', 'schema': 1,
                                                  'architecture': 'other'}, 'state_dict': {}}):
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    load_compact_checkpoint(payload)


if __name__ == '__main__':
    unittest.main()
