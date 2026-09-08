"""Verify saved numerical references without raw data, training or simulation."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def holm(p):
    order = np.argsort(p); out = np.empty(len(p))
    out[order] = np.minimum(1, np.maximum.accumulate(np.asarray(p)[order] * np.arange(len(p), 0, -1)))
    return out


def main():
    folder = ROOT / 'results/reference/benchmark/statistics'
    output = {}
    for name, expected in (('primary_tests', 12), ('uncertainty_tests', 6)):
        frame = pd.read_csv(folder / (name + '.csv'))
        assert len(frame) == expected
        assert (frame.permutations == 10000).all()
        assert np.allclose(holm(frame.raw_p.to_numpy()), frame.holm_p, atol=1e-12, rtol=1e-12)
        assert (frame.ci_95_lower <= frame.ci_95_upper).all()
        output[name] = {'contrasts': len(frame), 'holm_verified': True,
                        'holm_below_0_05': int((frame.holm_p < .05).sum())}
    width = pd.read_csv(ROOT / 'results/reference/width/statistics/exploratory_tests.csv')
    assert len(width) == 2
    assert np.allclose(holm(width.raw_p.to_numpy()), width.holm_p, atol=1e-12, rtol=1e-12)
    output['exploratory_width_tests'] = len(width)
    manifest = ROOT / 'provenance/REFERENCE_FILES.sha256'
    checked = 0
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            expected, relative = line.split('  ', 1)
            assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
            checked += 1
    output['reference_hashes_verified'] = checked
    print(json.dumps(output, indent=2))


if __name__ == '__main__': main()
