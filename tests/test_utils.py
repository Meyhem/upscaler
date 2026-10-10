import random

import numpy as np
import torch

from upscaler.utils import deep_merge, load_config, set_seed


def test_deep_merge_nested_and_does_not_mutate():
    base = {"a": 1, "b": {"c": 2, "d": 3}, "g": None}
    over = {"b": {"c": 20}, "e": 5, "g": {"x": 1}}
    out = deep_merge(base, over)
    assert out == {"a": 1, "b": {"c": 20, "d": 3}, "e": 5, "g": {"x": 1}}
    assert base == {"a": 1, "b": {"c": 2, "d": 3}, "g": None}


def test_load_config_resolves_base(tmp_path):
    (tmp_path / "base.yaml").write_text("seed: 1\ntrain:\n  lr: 0.1\n  iters: 10\n")
    (tmp_path / "child.yaml").write_text("_base_: base.yaml\ntrain:\n  lr: 0.5\n")
    cfg = load_config(tmp_path / "child.yaml")
    assert cfg == {"seed": 1, "train": {"lr": 0.5, "iters": 10}}


def test_set_seed_is_reproducible():
    set_seed(3)
    a = (random.random(), np.random.rand(), torch.rand(1).item())
    set_seed(3)
    b = (random.random(), np.random.rand(), torch.rand(1).item())
    assert a == b
