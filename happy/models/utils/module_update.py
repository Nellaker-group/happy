"""
Load pickled model .pt files saved with older PyTorch / PyTorch Geometric versions.

Problems this fixes:
1. nn.Module attributes added in newer PyTorch (e.g. _lazy_load_hook) are
   absent from old pickled module state, causing AttributeError during load.
2. torch_geometric internal module paths changed between versions (e.g.
   Inspector moved from torch_geometric.nn.conv.utils.inspector to
   torch_geometric.inspector in PyG 2.5), causing ModuleNotFoundError during load.

load_model() first loads the file normally. If that fails, it patches in safe
defaults for the missing attributes and aliases for the moved torch_geometric
paths, loads the model, and re-saves it against the current library versions
so the patches are not needed next time. The original file is kept in an
old_version/ dir next to it.

Usage (to update a file without running inference):
    python -m happy.models.utils.module_update /path/to/model.pt
"""

import os
import shutil
import sys
from pathlib import Path

import typer
import torch
import torch.nn as nn

# Attributes added to nn.Module / MessagePassing in newer PyTorch / PyG versions
# that are absent from old pickled module state. Accessing them falls through
# to __getattr__ and raises AttributeError without this patch.
_MISSING_ATTR_DEFAULTS = {
    "_lazy_load_hook": None,
    "_is_full_backward_hook": None,
    "_forward_pre_hooks_with_kwargs": {},
    "_forward_hooks_with_kwargs": {},
    "_forward_hooks_always_called": {},
    "_state_dict_pre_hooks": {},
    "_decomposed_layers": 1,
}
_patched = set()


def load_model(model_path, device="cpu"):
    """Load a pickled model, updating the file if it was saved with older versions."""
    model_path = Path(model_path)
    # attributes only used at runtime (e.g. in forward) can be missing even when
    # the file itself loads, so this patch is always applied
    _patch_module_getattr()
    try:
        return torch.load(model_path, map_location=device)
    except (AttributeError, ModuleNotFoundError) as e:
        print(
            f"Could not load {model_path} with the current PyTorch / PyTorch "
            f"Geometric versions ({type(e).__name__}: {e}). Updating it."
        )
    model = _load_with_patches(model_path)
    _save_updated(model, model_path)
    return model.to(device)


def _load_with_patches(model_path):
    _patch_pyg_module_paths()
    _patch_message_passing_setstate()
    model = torch.load(model_path, map_location="cpu")
    _inject_missing_attributes(model)
    return model


def _save_updated(model, model_path):
    """Re-save the updated model in place, keeping the original in old_version/."""
    backup_path = model_path.parent / "old_version" / model_path.name
    tmp_path = model_path.with_name(f"{model_path.name}.tmp")
    try:
        # never overwrite an existing backup: it holds the original file
        if not backup_path.exists():
            backup_path.parent.mkdir(exist_ok=True)
            shutil.copy2(model_path, backup_path)
        torch.save(model, tmp_path)
        os.replace(tmp_path, model_path)
    except OSError as e:
        tmp_path.unlink(missing_ok=True)
        print(f"Could not save the updated model ({e}). Using it for this run only.")
        print(
            f"To update the file, run with write access to its folder: "
            f"python -m happy.models.utils.module_update {model_path}"
        )
        return
    print(
        f"Updated {model_path} to the current PyTorch / PyTorch Geometric versions. "
        f"The original is saved at {backup_path}"
    )


def _patch_module_getattr():
    """Patch nn.Module.__getattr__ to return safe defaults for attributes added
    in newer PyTorch that are absent from old pickled module state."""
    if "getattr" in _patched:
        return
    _original = nn.Module.__getattr__

    def _patched_getattr(self, name: str):
        if name in _MISSING_ATTR_DEFAULTS:
            default = _MISSING_ATTR_DEFAULTS[name]
            object.__setattr__(self, name, default)
            return default
        return _original(self, name)

    nn.Module.__getattr__ = _patched_getattr
    _patched.add("getattr")


def _patch_pyg_module_paths():
    """Register sys.modules aliases for torch_geometric paths that moved between
    PyG versions. Models saved with older PyG embed the old paths in pickle."""
    try:
        import torch_geometric.inspector as _inspector
    except ImportError:
        # PyG < 2.5 still has the old paths
        return
    sys.modules.setdefault("torch_geometric.nn.conv.utils.inspector", _inspector)
    sys.modules.setdefault("torch_geometric.nn.conv.utils", _inspector)


def _patch_message_passing_setstate():
    """Patch MessagePassing.__setstate__ to discard the stale pickled Inspector
    and recreate it from the current class.

    The Inspector holds only metadata about a layer's method signatures — not
    learned weights. Between PyG versions its internal structure changed (e.g.
    self.cls → self._cls), so a pickled Inspector from an older PyG is
    incompatible with the new API. Recreating it fresh is safe.
    """
    if "setstate" in _patched:
        return
    try:
        from torch_geometric.inspector import Inspector
    except ImportError:
        # PyG < 2.5 can use the pickled Inspector as is
        return
    from torch_geometric.nn.conv.message_passing import MessagePassing

    _original = MessagePassing.__setstate__

    def _patched_setstate(self, data: dict):
        inspector = Inspector(self.__class__)
        inspector.inspect_signature(self.message)
        inspector.inspect_signature(self.aggregate, exclude=[0, 'aggr'])
        inspector.inspect_signature(self.message_and_aggregate, [0])
        inspector.inspect_signature(self.update, exclude=[0])
        inspector.inspect_signature(self.edge_update)
        data['inspector'] = inspector
        _original(self, data)

    MessagePassing.__setstate__ = _patched_setstate
    _patched.add("setstate")


def _inject_missing_attributes(model: nn.Module):
    """Walk every module and inject missing attributes directly into __dict__
    so they are serialised when the model is re-saved.

    The __getattr__ patch only fires when an attribute is accessed at runtime,
    so attributes only needed during forward() (e.g. decomposed_layers) are
    never set during a load-only pass. This function proactively sets them on
    every submodule.
    """
    for module in model.modules():
        for attr, default in _MISSING_ATTR_DEFAULTS.items():
            if attr not in module.__dict__:
                object.__setattr__(module, attr, default)


def main(
    model_path: Path = typer.Argument(..., help="Path to the .pt model file to update"),
):
    if not model_path.exists():
        print(f"Error: file not found: {model_path}")
        sys.exit(1)
    _patch_module_getattr()
    model = _load_with_patches(model_path)
    print(f"Loaded {type(model).__name__} from {model_path}")
    _save_updated(model, model_path)


if __name__ == "__main__":
    typer.run(main)
