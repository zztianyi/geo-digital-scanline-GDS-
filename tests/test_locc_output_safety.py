"""Read-only regression for the experiment's frozen-input output boundary."""
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts/99_experiments'))
from validate_locc_natural_gaps import validate_output_location, infer


class OutputSafetyTest(unittest.TestCase):
    def test_real_frozen_directories_rejected_before_any_write(self):
        prior = ROOT/'outputs/orthogonal_scanline_constraint/20260919_081148'
        manifest = prior/'manifest.json'
        if not manifest.exists():
            self.skipTest('Local frozen integration fixture is unavailable')
        original = hashlib.sha256(manifest.read_bytes()).hexdigest()
        baseline = Path(json.loads(manifest.read_text(encoding='utf-8'))['baseline'])
        for output in (prior, prior/'must_not_create', baseline, baseline/'must_not_create', prior.parent):
            with self.subTest(output=str(output)), self.assertRaises(ValueError):
                validate_output_location(prior, output)
        with self.assertRaises(ValueError):
            infer(prior, prior, 25)
        allowed = ROOT/'outputs/locc_validation/safety_check_not_created'
        validate_output_location(prior, allowed)
        self.assertFalse(allowed.exists())
        self.assertEqual(original, hashlib.sha256(manifest.read_bytes()).hexdigest())


if __name__ == '__main__':
    unittest.main()
