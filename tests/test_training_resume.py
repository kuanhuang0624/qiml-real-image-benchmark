from pathlib import Path

import numpy as np
import pytest
torch = pytest.importorskip("torch")

from qiml_benchmark.vqc.model import VQCDR
from qiml_benchmark.vqc.training import train


def test_segmented_resume_equivalence(tmp_path: Path) -> None:
    rng = np.random.default_rng(3); x = rng.normal(size=(8, 8)).astype(np.float32); y = np.arange(8) % 2
    direct = VQCDR(2, 42); train(direct, x, y, steps=4, learning_rate=.01)
    segmented = VQCDR(2, 42); checkpoint = tmp_path / "resume.pt"
    train(segmented, x, y, steps=2, learning_rate=.01, checkpoint=checkpoint)
    train(segmented, x, y, steps=4, learning_rate=.01, resume=checkpoint)
    for expected, actual in zip(direct.parameters(), segmented.parameters()): assert torch.allclose(expected, actual, atol=1e-7, rtol=1e-7)
