# ==================================
# Copyright: CEA-LIST/DIASI/SIALV/
# Author : Torchaug Developers
# License: CECILL-C
# ==================================

# Code partially based on Torchvision (BSD 3-Clause License), available at:
#   https://github.com/pytorch/vision

from __future__ import annotations

import functools
from typing import Callable, Dict, List, Optional, Sequence, Type, Union

import torch

from torchaug import ta_tensors


_FillType = Optional[Union[int, float, Sequence[int], Sequence[float]]]
_FillTypeJIT = Optional[List[float]]


# {functional: {input_type: type_specific_kernel}}
_KERNEL_REGISTRY: Dict[Callable, Dict[Type, Callable]] = {}


def _kernel_ta_tensor_wrapper(kernel):
    @functools.wraps(kernel)
    def wrapper(inpt, *args, **kwargs):
        # If you're wondering whether we could / should get rid of this wrapper,
        # the answer is no: we want to pass pure Tensors to avoid the overhead
        # of the __torch_function__ machinery. Note that this is always valid,
        # regardless of whether we override __torch_function__ in our base class
        # or not.
        # Also, even if we didn't call `as_subclass` here, we would still need
        # this wrapper to call wrap(), because the TVTensor type would be
        # lost after the first operation due to our own __torch_function__
        # logic.
        output = kernel(inpt.as_subclass(torch.Tensor), *args, **kwargs)
        return ta_tensors.wrap(output, like=inpt)

    return wrapper


def _register_kernel_internal(functional, input_type, *, ta_tensor_wrapper=True):
    registry = _KERNEL_REGISTRY.setdefault(functional, {})
    if input_type in registry:
        raise ValueError(f"Functional {functional} already has a kernel registered for type {input_type}.")

    def decorator(kernel):
        registry[input_type] = (
            _kernel_ta_tensor_wrapper(kernel)
            if issubclass(input_type, ta_tensors.TATensor) and ta_tensor_wrapper
            else kernel
        )
        return kernel

    return decorator


def _name_to_functional(name):
    import torchvision.transforms.v2.functional  # noqa

    import torchaug.transforms.functional  # noqa

    def get_torchaug_functional(name):
        try:
            return getattr(torchaug.transforms.functional, name)
        except AttributeError:
            return None

    def get_torchvision_functional(name):
        try:
            return getattr(torchvision.transforms.v2.functional, name)
        except AttributeError:
            return None

    functional = get_torchaug_functional(name) or get_torchvision_functional(name)

    if functional is None:
        raise ValueError(
            f"Could not find functional with name '{name}' in torchaug.transforms.functional or "
            "torchvision.transforms.v2.functional."
        )

    return functional


_BUILTIN_DATAPOINT_TYPES = {
    obj for obj in ta_tensors.__dict__.values() if isinstance(obj, type) and issubclass(obj, ta_tensors.TATensor)
}


def register_kernel(functional, ta_tensor_cls):
    """Decorate a kernel to register it for a functional and a (custom) ta_tensor type.

    See :ref:`sphx_glr_auto_examples_transforms_plot_custom_ta_tensors.py` for usage
    details.
    """
    if isinstance(functional, str):
        functional = _name_to_functional(name=functional)
    elif not (
        callable(functional)
        and getattr(functional, "__module__", "").startswith("torchvision.transforms.v2.functional")
    ):
        raise ValueError(
            f"Kernels can only be registered on functionals from the torchvision.transforms.v2.functional namespace, "
            f"but got {functional}."
        )

    if not (isinstance(ta_tensor_cls, type) and issubclass(ta_tensor_cls, ta_tensors.TATensor)):
        raise ValueError(
            f"Kernels can only be registered for subclasses of ~torchaug.ta_tensors.TATensor, but got {ta_tensor_cls}."
        )

    if ta_tensor_cls in _BUILTIN_DATAPOINT_TYPES:
        raise ValueError(f"Kernels cannot be registered for the builtin ta_tensor classes, but got {ta_tensor_cls}")

    return _register_kernel_internal(functional, ta_tensor_cls, ta_tensor_wrapper=False)


def _get_kernel(functional, input_type, *, allow_passthrough=False):
    registry = _KERNEL_REGISTRY.get(functional)
    if not registry:
        raise ValueError(f"No kernel registered for functional {functional.__name__}.")

    for cls in input_type.__mro__:
        if cls in registry:
            return registry[cls]
        elif cls is ta_tensors.TATensor:
            # We don't want user-defined ta_tensors to dispatch to the pure Tensor kernels, so we explicit stop the
            # MRO traversal before hitting torch.Tensor. We can even stop at ta_tensors.TATensor, since we don't
            # allow kernels to be registered for ta_tensors.TATensor anyway.
            break

    if allow_passthrough:
        return lambda inpt, *args, **kwargs: inpt

    raise TypeError(
        f"Functional F.{functional.__name__} supports inputs of type {registry.keys()}, but got {input_type} instead."
    )


# This basically replicates _register_kernel_internal, but with a specialized wrapper for five_crop / ten_crop
# We could get rid of this by letting _register_kernel_internal take arbitrary functionals rather
# than wrap_kernel: bool
def _register_five_ten_crop_kernel_internal(functional, input_type):
    registry = _KERNEL_REGISTRY.setdefault(functional, {})
    if input_type in registry:
        raise TypeError(f"Functional '{functional}' already has a kernel registered for type '{input_type}'.")

    def wrap(kernel):
        @functools.wraps(kernel)
        def wrapper(inpt, *args, **kwargs):
            output = kernel(inpt, *args, **kwargs)
            container_type = type(output)
            return container_type(ta_tensors.wrap(o, like=inpt) for o in output)

        return wrapper

    def decorator(kernel):
        registry[input_type] = wrap(kernel) if issubclass(input_type, ta_tensors.TATensor) else kernel
        return kernel

    return decorator
