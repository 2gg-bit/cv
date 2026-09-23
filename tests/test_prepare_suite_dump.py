"""Regression for unnamed MMCV configs; keep the formatter's exact output."""
import copy
import sys
from types import SimpleNamespace

import pytest

from test_dual_teacher_baseline import load_file

generator = load_file("suite_generator_test", "tools/prepare_ablation_suite.py")


def test_dump_uses_pretty_text_without_filename_dispatch(monkeypatch, tmp_path):
    values = dict(seed=678, labels=("ship",), model=dict(pg=None, enabled=False))
    before = copy.deepcopy(values)
    formatted = "seed = 678\nlabels = ('ship',)\nmodel = dict(pg=None, enabled=False)\n"

    class LegacyConfig:
        def __init__(self, data):
            assert data == before
            self.filename = None

        @property
        def pretty_text(self):
            return formatted

        def dump(self, path):
            self.filename.endswith(".py")  # MMCV 1.3.9's failing dispatch
    monkeypatch.setitem(sys.modules, "mmcv", SimpleNamespace(Config=LegacyConfig))
    with pytest.raises(AttributeError):
        LegacyConfig(values).dump(str(tmp_path / "broken.py"))
    path = tmp_path / "generated.py"
    generator.dump_config(values, path)
    assert path.read_bytes() == formatted.encode("utf-8")
    namespace = {}
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    assert {k: namespace[k] for k in values} == before
    assert values == before


def test_dump_does_not_hide_formatter_errors(monkeypatch, tmp_path):
    class InvalidConfig:
        def __init__(self, data):
            pass

        @property
        def pretty_text(self):
            raise ValueError("unserializable config")
    monkeypatch.setitem(sys.modules, "mmcv", SimpleNamespace(Config=InvalidConfig))
    target = tmp_path / "bad.py"
    with pytest.raises(ValueError, match="unserializable"):
        generator.dump_config({}, target)
    assert not target.exists()
