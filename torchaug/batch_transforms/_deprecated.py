import warnings

from ._color import BatchRandomGrayscale


class BatchRandomGrayScale(BatchRandomGrayscale):
    def __init__(
        self, p: float = 0.5, num_output_channels: int = 3, inplace: bool = False
    ) -> None:
        super().__init__(p, num_output_channels, inplace)
        warnings.warn(
            (
                "BatchRandomGrayScale changed its name to BatchRandomGrayscale, "
                "please change your import accordingly. BatchRandomGrayScale will be deleted in 0.4."
            ),
            category=DeprecationWarning,
            stacklevel=2,
        )
