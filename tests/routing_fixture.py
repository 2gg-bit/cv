"""Test the actual supervision call routing without MMDetection or a GPU."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_forward_class():
    source = ROOT / "ssod/models/dual_teacher.py"
    tree = ast.parse(source.read_text())
    dual = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DualTeacher")
    method = next(n for n in dual.body if isinstance(n, ast.FunctionDef) and n.name == "forward_train")
    fixture = ast.ClassDef(
        name="RoutingFixture", bases=[ast.Name(id="Base", ctx=ast.Load())],
        keywords=[], body=[method], decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[fixture], type_ignores=[]))

    class Base:
        def forward_train(self, *args, **kwargs):
            pass

    def split(data, key):
        return {
            tag: {k: (v[[i for i in range(len(data[key])) if data[key][i] == tag]] if torch.is_tensor(v)
                      else [v[i] for i in range(len(data[key])) if data[key][i] == tag])
                  for k, v in data.items()}
            for tag in set(data[key])
        }

    namespace = dict(Base=Base, torch=torch, dict_split=split, log_every_n=lambda *args: None,
                     weighted_loss=lambda loss, weight: {k: v * weight for k, v in loss.items()})
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["RoutingFixture"]
