from __future__ import annotations

import re

import pytest
import torch
import torchvision.transforms.functional as F_tv

import torchaug.transforms as transforms
import torchaug.transforms.functional as F

from ...base_tester import BaseTesterTransform


class TestRandomColorJitter(BaseTesterTransform):
    def test_functional_float(self):
        torch.manual_seed(28)
        tensor = self.get_float_image((3, 16, 16))

        out_jittered = transforms.RandomColorJitter(0.5, 0.3, 0.1, 0.3, 1.0)(tensor)
        transforms.RandomColorJitter(0.5, 0.3, 0.1, 0.3, 0.5)(tensor)
        out_not_jittered = transforms.RandomColorJitter(0.5, 0.3, 0.1, 0.3, 0.0)(tensor)

        brightness, contrast, saturation, hue = (
            torch.tensor(0.8414264917373657),
            torch.tensor(0.9322187900543213),
            torch.tensor(0.9402793049812317),
            torch.tensor(0.1355568766593933),
        )

        torchvision_out = F_tv.adjust_brightness(
            F_tv.adjust_hue(
                F_tv.adjust_contrast(
                    F_tv.adjust_saturation(tensor, saturation), contrast
                ),
                hue,
            ),
            brightness,
        )

        torch.testing.assert_close(out_jittered, torchvision_out)
        torch.testing.assert_close(out_not_jittered, tensor)

    def test_functional_uint8(self):
        torch.manual_seed(28)
        tensor = self.get_uint8_image((3, 16, 16))

        out_jittered = transforms.RandomColorJitter(0.5, 0.3, 0.1, 0.3, 1.0)(tensor)
        transforms.RandomColorJitter(0.5, 0.3, 0.1, 0.3, 0.5)(tensor)
        out_not_jittered = transforms.RandomColorJitter(0.5, 0.3, 0.1, 0.3, 0.0)(tensor)

        brightness, contrast, saturation, hue = (
            torch.tensor(0.8414264917373657),
            torch.tensor(0.9322187900543213),
            torch.tensor(0.9402793049812317),
            torch.tensor(0.1355568766593933),
        )

        torchvision_out = F_tv.adjust_brightness(
            F_tv.adjust_hue(
                F_tv.adjust_contrast(
                    F_tv.adjust_saturation(tensor, saturation), contrast
                ),
                hue,
            ),
            brightness,
        )

        torch.testing.assert_close(out_jittered, torchvision_out)
        torch.testing.assert_close(out_not_jittered, tensor)

    @pytest.mark.parametrize(
        "brightness,contrast,saturation,hue,p,repr",
        [
            (
                0.5,
                0.1,
                0.2,
                0.0,
                0.5,
                "RandomColorJitter(brightness=(0.5, 1.5), contrast=(0.9, 1.1), saturation=(0.8, 1.2), hue=None, p=0.5)",
            ),
            (
                0.0,
                0.1,
                0.2,
                0.1,
                1.0,
                "RandomColorJitter(brightness=None, contrast=(0.9, 1.1), saturation=(0.8, 1.2), hue=(-0.1, 0.1), p=1.0)",
            ),
        ],
    )
    def test_repr(
        self,
        brightness: float | tuple[float, float] | None,
        contrast: float | tuple[float, float] | None,
        saturation: float | tuple[float, float] | None,
        hue: float | tuple[float, float] | None,
        p: float,
        repr: str,
    ):
        assert (
            transforms.RandomColorJitter(
                brightness, contrast, saturation, hue, p
            ).__repr__()
            == repr
        )


class TestRandomSolarize(BaseTesterTransform):
    def test_functional_float(self):
        torch.manual_seed(28)

        tensor = self.get_float_image((3, 16, 16))

        out_not_solarized = transforms.RandomSolarize(0.5, 0.0)(tensor)
        transforms.RandomSolarize(0.5, 0.5)(tensor)
        out_solarized = transforms.RandomSolarize(0.5, 1.0)(tensor)

        expected_out = F.solarize(tensor, 0.5)

        torch.testing.assert_close(out_solarized, expected_out)
        torch.testing.assert_close(out_not_solarized, tensor)

    def test_functional_uint8(self):
        torch.manual_seed(28)

        tensor = self.get_uint8_image((3, 16, 16))

        out_not_solarized = transforms.RandomSolarize(128, 0.0)(tensor)
        transforms.RandomSolarize(128, 0.5)(tensor)
        out_solarized = transforms.RandomSolarize(128, 1.0)(tensor)

        expected_out = F.solarize(tensor, 128)

        torch.testing.assert_close(out_solarized, expected_out)
        torch.testing.assert_close(out_not_solarized, tensor)

    def test_repr(self):
        assert re.match(
            r"RandomSolarize\(threshold=0.\d+, p=0.5, value_check=False\)",
            transforms.RandomSolarize(0.5, 0.5, False).__repr__(),
        )
        assert re.match(
            r"RandomSolarize\(threshold=0.\d+, p=0.5, value_check=True\)",
            transforms.RandomSolarize(0.5, 0.5, True).__repr__(),
        )