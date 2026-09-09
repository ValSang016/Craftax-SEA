"""Small, dependency-light checkpoint helpers for Flax parameter PyTrees."""

from pathlib import Path

from flax import serialization


def save_params(path, params):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(serialization.to_bytes(params))


def load_params(path, template):
    return serialization.from_bytes(template, Path(path).read_bytes())


__all__ = ["load_params", "save_params"]
