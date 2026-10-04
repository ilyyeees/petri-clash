"""Portable NCA checkpoint I/O shared by training, evaluation, and play.

Only load checkpoint files from sources you trust. Loading always uses PyTorch's
restricted ``weights_only`` unpickler; arbitrary Python objects are unsupported.
"""

from collections import OrderedDict
from collections.abc import Mapping
import copy
import json
import os
from pathlib import Path
import pickle
import tempfile

import numpy as np
import torch


def _plain_key(key):
    if not isinstance(key, str):
        raise TypeError("state dict keys must be strings")
    return ".".join(part for part in key.split(".") if part != "_orig_mod")


def normalize_state_dict(state_dict):
    """Strip compile wrappers per key, including mixed and nested wrappers.

    Ambiguous names are rejected instead of silently overwriting parameters.
    Module metadata is kept, preferring the wrapped module over its wrapper.
    """
    if not isinstance(state_dict, Mapping):
        raise TypeError("checkpoint model must be a state dict")
    normalized = OrderedDict()
    for key, value in state_dict.items():
        plain_key = _plain_key(key)
        if plain_key in normalized:
            raise ValueError(f"duplicate checkpoint parameter after normalization: {plain_key}")
        normalized[plain_key] = value
    metadata = getattr(state_dict, "_metadata", None)
    if metadata is not None:
        if not isinstance(metadata, Mapping):
            raise ValueError("state dict metadata must be a mapping")
        for key, value in metadata.items():
            _plain_key(key)  # Validate names before sorting compile wrappers.
            if not isinstance(value, Mapping):
                raise ValueError("state dict module metadata must be a mapping")
        normalized._metadata = OrderedDict()
        for key in sorted(metadata, key=lambda name: name.count("_orig_mod")):
            normalized._metadata[_plain_key(key)] = copy.deepcopy(metadata[key])
    return normalized


def load_model_state(model, state_dict, *, strict=True):
    """Load plain or compiled weights into a plain or compiled model."""
    normalized = normalize_state_dict(state_dict)
    target = model.state_dict()
    # Map canonical names back to the destination wrapper structure. This also
    # supports individually compiled child modules, not only whole-model wraps.
    normalize_state_dict(target)  # Check for ambiguous destination names too.
    target_names = {_plain_key(key): key for key in target}
    remapped = OrderedDict((target_names.get(key, key), value) for key, value in normalized.items())
    if hasattr(normalized, "_metadata"):
        remapped._metadata = OrderedDict()
        for key in getattr(target, "_metadata", {}):
            plain_key = _plain_key(key)
            if plain_key in normalized._metadata:
                remapped._metadata[key] = normalized._metadata[plain_key]
    return model.load_state_dict(remapped, strict=strict)


def _load_legacy_numpy_checkpoint(path, map_location):
    # Old resumable checkpoints stored NumPy's uint32 RNG array directly. Keep
    # support for this exact representation without unrestricted pickle loading.
    numpy_core = np._core if hasattr(np, "_core") else np.core
    reconstruct = numpy_core.multiarray._reconstruct
    safe_globals = [
        (reconstruct, "numpy.core.multiarray._reconstruct"),
        (reconstruct, "numpy._core.multiarray._reconstruct"),
        np.ndarray,
        np.dtype,
        type(np.dtype(np.uint32)),
    ]
    with torch.serialization.safe_globals(safe_globals):
        return torch.load(path, map_location=map_location, weights_only=True)


def load_checkpoint_file(path, map_location="cpu"):
    """Read a checkpoint safely, including legacy NumPy RNG snapshots.

    There is deliberately no automatic ``weights_only=False`` fallback.
    """
    try:
        try:
            blob = torch.load(path, map_location=map_location, weights_only=True)
        except pickle.UnpicklingError as exc:
            if "numpy" not in str(exc).lower():
                raise
            blob = _load_legacy_numpy_checkpoint(path, map_location)
    except (EOFError, pickle.UnpicklingError, IndexError) as exc:
        # Restricted unpickling can report truncated/unsupported input through
        # several exception types. Keep one recoverable input-error boundary,
        # including the legacy attempt, without hiding I/O or resource errors.
        raise ValueError(f"invalid or unsupported NCA checkpoint: {path}") from exc
    if not isinstance(blob, Mapping) or "model" not in blob:
        raise ValueError(f"not an NCA checkpoint (missing model state): {path}")
    if not isinstance(blob["model"], Mapping):
        raise ValueError(f"invalid model state in checkpoint: {path}")
    return blob


def checkpoint_config(blob, fallback=None, *, checkpoint_path=None):
    """Merge defaults < neighboring resolved_config.json < saved config.

    All values are copied. Saved architecture, data layout, and evaluation
    settings win over defaults. Callers can apply explicit runtime overrides
    (for example, the device) to the returned dict without modifying the input.
    """
    def merge(base, extra):
        result = copy.deepcopy(dict(base))
        for key, value in extra.items():
            if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
                result[key] = merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        return result

    result = merge({}, fallback or {})
    if checkpoint_path is not None:
        resolved_path = Path(checkpoint_path).parent.parent / "resolved_config.json"
        if resolved_path.is_file():
            resolved = json.loads(resolved_path.read_text())
            if not isinstance(resolved, Mapping):
                raise ValueError(f"invalid checkpoint config: {resolved_path}")
            result = merge(result, resolved)
    saved = blob.get("config")
    if saved is not None:
        if not isinstance(saved, Mapping):
            raise ValueError("checkpoint config must be a mapping")
        result = merge(result, saved)
    if not result:
        raise ValueError("checkpoint has no saved config or fallback config")
    return result


def portable_rng_blob(blob):
    """Encode legacy NumPy RNG arrays as primitives accepted by weights_only."""
    result = dict(blob)
    if "numpy" in result:
        name, keys, pos, has_gauss, cached_gaussian = result["numpy"]
        result["numpy"] = (name, keys.tolist() if hasattr(keys, "tolist") else list(keys),
                           int(pos), int(has_gauss), float(cached_gaussian))
    return result


def save_checkpoint_file(blob, path):
    """Atomically replace a checkpoint, preserving the old file on failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(blob, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
