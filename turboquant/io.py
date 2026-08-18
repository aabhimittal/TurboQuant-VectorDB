"""Saving and loading indexes and quantizers, without pickle.

An index that cannot outlive its process is a demo. Training the
codebooks is the expensive part -- minutes of k-means over millions of
vectors -- and it has to survive a deploy.

Three decisions shape this module:

**No pickle.** `np.load(..., allow_pickle=True)` executes arbitrary code
in the file being read, which turns "restore the index from object
storage" into remote code execution if anyone can write to that bucket.
Everything here is plain arrays plus a JSON manifest, and loading passes
`allow_pickle=False` explicitly. The cost is that we cannot serialize
arbitrary Python objects -- which is fine, because we do not have any.

**Reflective, not hand-written.** Each class would otherwise need a
`to_dict`/`from_dict` pair, and the failure mode of that pattern is
silent: someone adds an attribute, forgets the serializer, and the
reloaded index answers queries slightly wrong forever. Instead we walk
`__dict__` and refuse to guess -- an attribute of an unrecognized type
raises at *save* time, so a new field cannot slip through unnoticed.

**Derived state is recomputed, not stored.** An IVF index's inverted
lists are a permutation of its cell assignments, and they cost more than
the codes they index (8 bytes per vector of int64 ids, against 8-32 bytes
of actual code). We drop them on save and rebuild on load. Same for the
adaptive quantizer's bit-packing layout, which is a pure function of its
bit allocation.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import numpy as np

from .index.flat import FlatIndex, QuantizedFlatIndex
from .index.ivfpq import IVFPQIndex
from .index.turbo import TurboIndex
from .quantizers.adaptive import AdaptiveBitQuantizer
from .quantizers.anisotropic import AnisotropicPQ
from .quantizers.opq import OPQProductQuantizer
from .quantizers.product import ProductQuantizer
from .quantizers.scalar import ScalarQuantizer

FORMAT = "turboquant"
FORMAT_VERSION = 1

# Every class this module is willing to reconstruct. Loading refuses
# anything not on this list, so a tampered manifest cannot name an
# arbitrary importable class.
_REGISTRY: dict[str, type] = {
    cls.__name__: cls
    for cls in (
        ScalarQuantizer,
        AdaptiveBitQuantizer,
        ProductQuantizer,
        AnisotropicPQ,
        OPQProductQuantizer,
        FlatIndex,
        QuantizedFlatIndex,
        IVFPQIndex,
        TurboIndex,
    )
}

# Attributes that are a pure function of other saved state. Skipped on
# save, recomputed by the matching hook on load.
_DERIVED: dict[str, set[str]] = {
    "AdaptiveBitQuantizer": {"_groups", "_stream_bits"},
    "IVFPQIndex": {"list_ids"},
}

_JSON_SCALARS = (bool, int, float, str, type(None))


def save(obj: Any, path: str | pathlib.Path) -> pathlib.Path:
    """Write a quantizer or index to `path` (an .npz file).

    Raises:
        TypeError: `obj`'s class is not serializable, or it holds an
            attribute of a type this module does not know how to store.
    """
    path = pathlib.Path(path)
    arrays: dict[str, np.ndarray] = {}
    manifest = {
        "format": FORMAT,
        "version": FORMAT_VERSION,
        "root": _encode(obj, prefix="", arrays=arrays),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        __manifest__=np.frombuffer(
            json.dumps(manifest).encode("utf-8"), dtype=np.uint8
        ),
        **arrays,
    )
    # np.savez appends .npz when the name lacks it; report the real path.
    return path if path.suffix == ".npz" else path.with_suffix(path.suffix + ".npz")


def load(path: str | pathlib.Path) -> Any:
    """Read back an object written by `save`.

    Raises:
        ValueError: the file is not a TurboQuant archive, or was written
            by an incompatible format version.
        TypeError: the manifest names a class that is not serializable.
    """
    path = pathlib.Path(path)
    # allow_pickle stays False: see the module docstring.
    with np.load(path, allow_pickle=False) as data:
        if "__manifest__" not in data:
            raise ValueError(
                f"{path} is not a TurboQuant archive (no manifest). It may be "
                "a plain .npz or a truncated file."
            )
        manifest = json.loads(bytes(data["__manifest__"]).decode("utf-8"))
        if manifest.get("format") != FORMAT:
            raise ValueError(
                f"{path} has format {manifest.get('format')!r}, expected {FORMAT!r}"
            )
        version = manifest.get("version")
        if version != FORMAT_VERSION:
            raise ValueError(
                f"{path} was written in format version {version}, but this "
                f"build reads version {FORMAT_VERSION}. Re-save it with the "
                "version of TurboQuant that wrote it."
            )
        return _decode(manifest["root"], data)


# -------------------------------------------------------------------- encode
def _encode(obj: Any, prefix: str, arrays: dict[str, np.ndarray]) -> dict:
    """Recursively describe `obj`, appending its arrays to `arrays`."""
    name = type(obj).__name__
    if name not in _REGISTRY:
        raise TypeError(
            f"{name} is not serializable. Register it in turboquant.io._REGISTRY "
            "if it should be."
        )
    node: dict[str, Any] = {
        "__class__": name,
        "scalars": {},
        "arrays": {},
        "children": {},
        "array_lists": {},
    }
    derived = _DERIVED.get(name, set())

    for attr, value in vars(obj).items():
        if attr in derived:
            continue
        if isinstance(value, np.ndarray):
            key = f"{prefix}{attr}"
            arrays[key] = value
            node["arrays"][attr] = key
        elif isinstance(value, np.generic):
            # A 0-d numpy scalar; store as the equivalent Python value.
            node["scalars"][attr] = value.item()
        elif isinstance(value, _JSON_SCALARS):
            node["scalars"][attr] = value
        elif type(value).__name__ in _REGISTRY:
            node["children"][attr] = _encode(value, f"{prefix}{attr}.", arrays)
        elif isinstance(value, list) and all(
            isinstance(v, np.ndarray) for v in value
        ):
            keys = []
            for i, v in enumerate(value):
                key = f"{prefix}{attr}[{i}]"
                arrays[key] = v
                keys.append(key)
            node["array_lists"][attr] = keys
        else:
            raise TypeError(
                f"cannot serialize attribute {name}.{attr} of type "
                f"{type(value).__name__}. Add it to _DERIVED if it can be "
                "recomputed, or extend _encode to handle it."
            )
    return node


# -------------------------------------------------------------------- decode
def _decode(node: dict, data: np.lib.npyio.NpzFile) -> Any:
    name = node["__class__"]
    if name not in _REGISTRY:
        raise TypeError(f"manifest names unknown class {name!r}")
    cls = _REGISTRY[name]
    # Bypass __init__: every attribute is about to be restored verbatim,
    # and re-running the constructor would recompute things like
    # TurboIndex's budget split that the saved state already fixes.
    obj = cls.__new__(cls)

    for attr, value in node["scalars"].items():
        setattr(obj, attr, value)
    for attr, key in node["arrays"].items():
        setattr(obj, attr, data[key])
    for attr, keys in node.get("array_lists", {}).items():
        setattr(obj, attr, [data[key] for key in keys])
    for attr, child in node["children"].items():
        setattr(obj, attr, _decode(child, data))

    _rebuild_derived(obj)
    return obj


def _rebuild_derived(obj: Any) -> None:
    """Recompute whatever `_DERIVED` told us not to store."""
    if isinstance(obj, AdaptiveBitQuantizer):
        # An untrained quantizer has no allocation to lay out yet.
        if getattr(obj, "bits_per_dim", None) is not None:
            obj._build_layout()
    elif isinstance(obj, IVFPQIndex):
        obj.list_ids = [np.empty(0, dtype=np.int64) for _ in range(obj.nlist)]
        if obj.cells is not None and obj.cells.size:
            obj._file_ids(np.arange(obj.cells.shape[0], dtype=np.int64), obj.cells)
