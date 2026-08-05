"""
Checkpoint I/O and parameter freezing for LWTNet.

Two kinds of checkpoint are handled:

* **weights only** - the released AVObjects `.pt` file, or any `.pth` we write.
  Loaded leniently with `load_model_params`: keys that are missing (e.g. the
  adapters, when starting from a pretrained plain LWTNet) keep their
  initialisation, and a `module.` prefix from a DDP checkpoint is stripped.
* **training state** - `{model,optimizer,scheduler}_state_dict` plus the step /
  epoch counters, written by `save_training_state` for resuming a run.
"""

import os

import torch
from torch.nn.modules.batchnorm import _BatchNorm


def unwrap(model):
    """Return the underlying module of a DataParallel / DDP wrapped model."""
    return model.module if hasattr(model, 'module') else model


def _torch_load(path, map_location=None):
    return torch.load(path, map_location=map_location or 'cpu', weights_only=False)


def _extract_state_dict(obj):
    """Accept a raw state dict or a training checkpoint and return the weights."""
    if isinstance(obj, dict):
        for key in ('model_state_dict', 'state_dict', 'model'):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    return obj


def _strip_module_prefix(key):
    return key[len('module.'):] if key.startswith('module.') else key


def load_model_params(model, path, map_location=None, verbose=True):
    """
    Copy whatever of `path` fits into `model`, leaving the rest untouched.

    Returns the list of parameter names that were *not* found in the checkpoint,
    which is how a from-pretrained fine-tune reports its freshly initialised
    layers (the temporal adapters).
    """
    net = unwrap(model)
    loaded_state = _extract_state_dict(_torch_load(path, map_location))
    self_state = net.state_dict()

    copied = set()
    for name, param in loaded_state.items():
        key = name if name in self_state else _strip_module_prefix(name)

        if key not in self_state:
            if verbose:
                print(f"[checkpoint] {name} is not in the model, skipped")
            continue

        if self_state[key].size() != param.size():
            if self_state[key].numel() == param.numel():
                if verbose:
                    print(f"[checkpoint] reshaping {name}: "
                          f"model {tuple(self_state[key].shape)}, loaded {tuple(param.shape)}")
                param = param.reshape(self_state[key].shape)
            else:
                if verbose:
                    print(f"[checkpoint] size mismatch for {name}: "
                          f"model {tuple(self_state[key].shape)}, loaded {tuple(param.shape)}, skipped")
                continue

        self_state[key].copy_(param)
        copied.add(key)

    missing = [k for k in self_state if k not in copied]
    if verbose:
        print(f"[checkpoint] loaded {len(copied)}/{len(self_state)} tensors from {path}")
        if missing:
            print(f"[checkpoint] kept initialisation for {len(missing)} tensors, "
                  f"e.g. {missing[:4]}")
    return missing


def load_checkpoint(chkpt, model):
    """Backwards-compatible entry point used by the inference scripts."""
    load_model_params(model, chkpt)
    print("Checkpoint {} loaded!".format(chkpt))


def save_training_state(path, model, optimizer, scheduler, global_step,
                        global_epoch, config=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(
        {
            'model_state_dict': unwrap(model).state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'global_step': global_step,
            'global_epoch': global_epoch,
            'config': config,
        }, path)
    return path


def load_training_state(path, model, optimizer=None, scheduler=None,
                        map_location=None, device=None):
    """Restore a run in place and return its (global_step, global_epoch)."""
    checkpoint = _torch_load(path, map_location)

    if 'model_state_dict' not in checkpoint:
        raise ValueError(
            f"{path} holds weights only, not a training state. Start from it with "
            f"model.pretrained instead of checkpoint.resume.")

    unwrap(model).load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if device is not None:
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint['global_step'], checkpoint['global_epoch']


def freeze_except(model, trainable_modules):
    """
    Freeze the whole network except the named submodules.

    `trainable_modules` is a list of attribute paths on the model, e.g.
    ``["ff_lip", "ff_aud", "vid_adapter", "aud_adapter"]`` for an adapter-only
    fine-tune. An empty list (or None) trains everything.
    """
    net = unwrap(model)

    if not trainable_modules:
        for param in net.parameters():
            param.requires_grad = True
        return

    for param in net.parameters():
        param.requires_grad = False

    for name in trainable_modules:
        # Raises AttributeError with the offending name if it does not exist,
        # which beats silently training nothing.
        for param in net.get_submodule(name).parameters():
            param.requires_grad = True


def set_train_mode(model, frozen_bn_eval=True):
    """
    Put the model in training mode, but keep frozen BatchNorm layers in eval.

    Without this a partial fine-tune still drifts the running statistics of the
    frozen backbone, which quietly changes the encoder it was supposed to leave
    alone.
    """
    model.train()
    if not frozen_bn_eval:
        return
    for module in unwrap(model).modules():
        if isinstance(module, _BatchNorm) and not any(
                p.requires_grad for p in module.parameters(recurse=False)):
            module.eval()


def count_parameters(model):
    net = unwrap(model)
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    total = sum(p.numel() for p in net.parameters())
    return trainable, total
