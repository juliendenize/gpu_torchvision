# ==================================
# Copyright: CEA-LIST/DIASI/SIALV/
# Author : Torchaug Developers
# License: CECILL-C
# ==================================

# Code partially based on Torchvision (BSD 3-Clause License), available at:
#   https://github.com/pytorch/vision

import math
import numbers
import warnings
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

import torch
from torch.nn.functional import one_hot
from torch.utils._pytree import tree_flatten, tree_unflatten
from torchvision.transforms.v2._utils import _check_sequence_input, _parse_labels_getter, has_any, query_chw

from torchaug import ta_tensors

from . import functional as F
from ._transform import RandomApplyTransform, Transform
from ._utils import is_pure_tensor, query_size


class RandomErasing(RandomApplyTransform):
    """Randomly select a rectangle region in the input image or video and erase its pixels.

    'Random Erasing Data Augmentation' by Zhong et al. See https://arxiv.org/abs/1708.04896

    Args:
        p: probability that the random erasing operation will be performed.
        scale: range of proportion of erased area against input image.
        ratio: range of aspect ratio of erased area.
        value: erasing value. If a single int, it is used to
            erase all pixels. If a tuple of length 3, it is used to erase
            R, G, B channels respectively.
            If a str of 'random', erasing each pixel with random values.
        batch_inplace: whether to apply the batch transform in-place.
            Does not prevent functionals to make copy but can reduce time and memory consumption.
        num_chunks: number of chunks to split the batched input into.
        permute_chunks: whether to permute the chunks.
        batch_transform: whether to apply the transform in batch mode.
    """

    def __init__(
        self,
        p: float = 0.5,
        scale: Tuple[float, float] = (0.02, 0.33),
        ratio: Tuple[float, float] = (0.3, 3.3),
        value: float = 0.0,
        inplace: bool = False,
        batch_inplace: bool = False,
        num_chunks: int = 1,
        permute_chunks: bool = False,
        batch_transform: bool = False,
    ):
        (
            super().__init__(
                p=p,
                batch_inplace=batch_inplace,
                num_chunks=num_chunks,
                permute_chunks=permute_chunks,
                batch_transform=batch_transform,
            ),
        )
        if not isinstance(value, (numbers.Number, str, tuple, list)):
            raise TypeError("Argument value should be either a number or str or a sequence")
        if isinstance(value, str) and value != "random":
            raise ValueError("If value is str, it should be 'random'")
        if not isinstance(scale, (tuple, list)):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, (tuple, list)):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")
        if scale[0] < 0 or scale[1] > 1:
            raise ValueError("Scale should be between 0 and 1")
        self.scale = scale
        self.ratio = ratio
        if isinstance(value, (int, float)):
            self.value = [float(value)]
        elif isinstance(value, str):
            self.value = None
        elif isinstance(value, (list, tuple)):
            self.value = [float(v) for v in value]
        else:
            self.value = value
        self.inplace = inplace

        self._log_ratio = torch.log(torch.tensor(self.ratio))

    def _call_kernel(self, functional: Callable, inpt: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(
            inpt,
            (
                ta_tensors.BoundingBoxes,
                ta_tensors.BatchBoundingBoxes,
                ta_tensors.Mask,
                ta_tensors.BatchMasks,
            ),
        ):
            warnings.warn(
                f"{type(self).__name__}() is currently passing through inputs of type "
                f"ta_tensors.{type(inpt).__name__}. This will likely change in the future."
            )
        return super()._call_kernel(functional, inpt, *args, **kwargs)

    def _get_params(
        self,
        flat_inputs: List[Any],
        num_chunks: int,
        chunks_indices: Tuple[torch.Tensor, ...],
    ) -> List[Dict[str, Any]]:
        img_c, img_h, img_w = query_chw(flat_inputs)

        if self.value is not None and len(self.value) not in (1, img_c):
            raise ValueError(
                f"If value is a sequence, it should have either a single value or {img_c} (number of inpt channels)"
            )

        area = img_h * img_w

        log_ratio = self._log_ratio
        params = []

        for i in range(num_chunks):
            for _ in range(10):
                erase_area = area * torch.empty(1).uniform_(self.scale[0], self.scale[1]).item()
                aspect_ratio = torch.exp(
                    torch.empty(1).uniform_(
                        log_ratio[0],  # type: ignore[arg-type]
                        log_ratio[1],  # type: ignore[arg-type]
                    )
                ).item()

                h = int(round(math.sqrt(erase_area * aspect_ratio)))
                w = int(round(math.sqrt(erase_area / aspect_ratio)))
                if not (h < img_h and w < img_w):
                    continue

                if self.value is None:
                    v = torch.empty([img_c, h, w], dtype=torch.float32).normal_()
                else:
                    v = torch.tensor(self.value)[:, None, None]

                i = torch.randint(0, img_h - h + 1, size=(1,)).item()
                j = torch.randint(0, img_w - w + 1, size=(1,)).item()
                break
            else:
                i, j, h, w, v = 0, 0, img_h, img_w, None

            params.append({"i": i, "j": j, "h": h, "w": w, "v": v})

        return params

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if params["v"] is not None:
            inpt = self._call_kernel(
                F.erase,
                inpt,
                **params,
                inplace=self.inplace,
            )

        return inpt


class JPEG(Transform):
    """Apply JPEG compression and decompression to the given images.
    The input is expected to be of a tensor of dtype uint8, on CPU, and have [..., 3 or 1, H, W] shape,
    where ... means an arbitrary number of leading dimensions.

    Args:
        quality: JPEG quality, from 1 to 100. Lower means more compression.
            If quality is a sequence like (min, max), it specifies the range of JPEG quality to
            randomly select from (inclusive of both ends).

    Returns:
        image with JPEG compression.
    """

    def __init__(self, quality: Union[int, Sequence[int]]):
        super().__init__()

        if isinstance(quality, int):
            quality = [quality, quality]
        else:
            _check_sequence_input(quality, "quality", req_sizes=(2,))

        if not (1 <= quality[0] <= quality[1] <= 100 and isinstance(quality[0], int) and isinstance(quality[1], int)):
            raise ValueError(f"quality must be an integer from 1 to 100, got {quality =}")

        self.quality = quality

    def _get_params(
        self,
        flat_inputs: List[Any],
        num_chunks: int,
        chunks_indices: Tuple[torch.Tensor, ...],
    ) -> List[Dict[str, Any]]:
        params: List[Dict[str, Any]] = []
        for i in range(num_chunks):
            quality = torch.randint(self.quality[0], self.quality[1] + 1, ()).item()
            params.append({"quality": quality})
        return params

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._call_kernel(F.jpeg, inpt, quality=params["quality"])


class _BaseMixUpCutMix(Transform):
    def __init__(self, *, alpha: float = 1.0, num_classes: int, labels_getter="default") -> None:
        super().__init__(batch_transform=True)
        self.alpha = float(alpha)
        self._dist = torch.distributions.Beta(torch.tensor([alpha]), torch.tensor([alpha]))

        self.num_classes = num_classes

        self._labels_getter = _parse_labels_getter(labels_getter)

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        flat_inputs, spec = tree_flatten(inputs)
        needs_transform_list = self._needs_transform_list(flat_inputs)

        if has_any(
            flat_inputs,
            ta_tensors.Image,
            ta_tensors.Video,
            ta_tensors.BoundingBoxes,
            ta_tensors.BatchBoundingBoxes,
            ta_tensors.Mask,
            ta_tensors.BatchMasks,
        ):
            raise ValueError(f"{type(self).__name__}() supports only batch of images or videos.")

        labels = self._labels_getter(inputs)
        if not isinstance(labels, torch.Tensor):
            raise ValueError(f"The labels must be a tensor, but got {type(labels)} instead.")
        elif labels.ndim != 1:
            raise ValueError(f"labels tensor should be of shape (batch_size,) but got shape {labels.shape} instead.")

        params = {
            "labels": labels,
            "batch_size": labels.shape[0],
            **self._get_params(
                [inpt for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list) if needs_transform],
                1,
                [torch.arange(labels.shape[0], device=labels.device)],
            )[0],
        }

        # By default, the labels will be False inside needs_transform_list, since they are a torch.Tensor coming
        # after an image or video. However, we need to handle them in _transform, so we make sure to set them to True
        needs_transform_list[next(idx for idx, inpt in enumerate(flat_inputs) if inpt is labels)] = True
        flat_outputs = [
            self._transform(inpt, params) if needs_transform else inpt
            for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list)
        ]

        return tree_unflatten(flat_outputs, spec)

    def _check_image_or_video(self, inpt: torch.Tensor, *, batch_size: int):
        expected_num_dims = 5 if isinstance(inpt, ta_tensors.BatchVideos) else 4
        if inpt.ndim != expected_num_dims:
            raise ValueError(
                f"Expected a batched input with {expected_num_dims} dims, but got {inpt.ndim} dimensions instead."
            )
        if inpt.shape[0] != batch_size:
            raise ValueError(
                f"The batch size of the image or video does not match the batch size of the labels: "
                f"{inpt.shape[0]} != {batch_size}."
            )

    def _mixup_label(self, label: torch.Tensor, *, lam: float) -> torch.Tensor:
        label = one_hot(label, num_classes=self.num_classes)
        if not label.dtype.is_floating_point:
            label = label.float()
        return label.roll(1, 0).mul_(1.0 - lam).add_(label.mul(lam))


class MixUp(_BaseMixUpCutMix):
    """Apply MixUp to the provided batch of images and labels.

    Paper: `mixup: Beyond Empirical Risk Minimization <https://arxiv.org/abs/1710.09412>`_.

    .. note::
        This transform is meant to be used on **batches** of samples, not
        individual images.
        The sample pairing is deterministic and done by matching consecutive
        samples in the batch, so the batch needs to be shuffled (this is an
        implementation detail, not a guaranteed convention.)

    In the input, the labels are expected to be a tensor of shape ``(batch_size,)``. They will be transformed
    into a tensor of shape ``(batch_size, num_classes)``.

    Args:
        alpha: hyperparameter of the Beta distribution used for mixup.
        num_classes: number of classes in the batch. Used for one-hot-encoding.
        labels_getter: indicates how to identify the labels in the input.
            By default, this will pick the second parameter as the labels if it's a tensor. This covers the most
            common scenario where this transform is called as ``MixUp()(imgs_batch, labels_batch)``.
            It can also be a callable that takes the same input as the transform, and returns the labels.
    """

    def _get_params(
        self, flat_inputs: List[Any], num_chunks: int, chunks_indices: Tuple[torch.Tensor, ...]
    ) -> List[Dict[str, Any]]:
        return [{"lam": float(self._dist.sample(()))}]

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        lam = params["lam"]

        if inpt is params["labels"]:
            return self._mixup_label(inpt, lam=lam)
        elif isinstance(
            inpt,
            (
                ta_tensors.BatchImages,
                ta_tensors.BatchVideos,
            ),
        ) or is_pure_tensor(inpt):
            self._check_image_or_video(inpt, batch_size=params["batch_size"])

            output = inpt.roll(1, 0).mul_(1.0 - lam).add_(inpt.mul(lam))

            if isinstance(
                inpt,
                (
                    ta_tensors.BatchImages,
                    ta_tensors.BatchVideos,
                ),
            ):
                output = ta_tensors.wrap(output, like=inpt)

            return output
        else:
            return inpt


class CutMix(_BaseMixUpCutMix):
    """Apply CutMix to the provided batch of images and labels.

    Paper: `CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features
    <https://arxiv.org/abs/1905.04899>`_.

    .. note::
        This transform is meant to be used on **batches** of samples, not
        individual images.
        The sample pairing is deterministic and done by matching consecutive
        samples in the batch, so the batch needs to be shuffled (this is an
        implementation detail, not a guaranteed convention.)

    In the input, the labels are expected to be a tensor of shape ``(batch_size,)``. They will be transformed
    into a tensor of shape ``(batch_size, num_classes)``.

    Args:
        alpha: hyperparameter of the Beta distribution used for mixup.
        num_classes: number of classes in the batch. Used for one-hot-encoding.
        labels_getter: indicates how to identify the labels in the input.
            By default, this will pick the second parameter as the labels if it's a tensor. This covers the most
            common scenario where this transform is called as ``CutMix()(imgs_batch, labels_batch)``.
            It can also be a callable that takes the same input as the transform, and returns the labels.
    """

    def _get_params(
        self, flat_inputs: List[Any], num_chunks: int, chunks_indices: Tuple[torch.Tensor, ...]
    ) -> List[Dict[str, Any]]:
        lam = float(self._dist.sample(()))  # type: ignore[arg-type]

        H, W = query_size(flat_inputs)

        r_x = torch.randint(W, size=(1,))
        r_y = torch.randint(H, size=(1,))

        r = 0.5 * math.sqrt(1.0 - lam)
        r_w_half = int(r * W)
        r_h_half = int(r * H)

        x1 = int(torch.clamp(r_x - r_w_half, min=0))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        x2 = int(torch.clamp(r_x + r_w_half, max=W))
        y2 = int(torch.clamp(r_y + r_h_half, max=H))
        box = (x1, y1, x2, y2)

        lam_adjusted = float(1.0 - (x2 - x1) * (y2 - y1) / (W * H))

        return [{"box": box, "lam_adjusted": lam_adjusted}]

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if inpt is params["labels"]:
            return self._mixup_label(inpt, lam=params["lam_adjusted"])
        elif isinstance(
            inpt,
            (
                ta_tensors.BatchImages,
                ta_tensors.BatchVideos,
            ),
        ) or is_pure_tensor(inpt):
            self._check_image_or_video(inpt, batch_size=params["batch_size"])

            x1, y1, x2, y2 = params["box"]
            rolled = inpt.roll(1, 0)
            output = inpt.clone()
            output[..., y1:y2, x1:x2] = rolled[..., y1:y2, x1:x2]

            if isinstance(
                inpt,
                (
                    ta_tensors.Image,
                    ta_tensors.Video,
                    ta_tensors.BatchImages,
                    ta_tensors.BatchVideos,
                ),
            ):
                output = ta_tensors.wrap(output, like=inpt)

            return output
        else:
            return inpt
