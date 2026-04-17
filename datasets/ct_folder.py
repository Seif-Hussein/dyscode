from pathlib import Path

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

from .image_base import ImageBaseDataset
from .registry import register_dataset


@register_dataset(name='CTFolder')
class CTFolderDataset(ImageBaseDataset):
    def __init__(self,
                 image_root_path: str,
                 valid_extensions: list,
                 resolution: int,
                 start_idx,
                 end_idx,
                 device,
                 center_crop: bool = True,
                 value_min=None,
                 value_max=None,
                 percentile_min=None,
                 percentile_max=None):
        super().__init__()

        self.valid_extensions = valid_extensions
        self.resolution = resolution
        self.device = device
        self.image_root_path = image_root_path
        self.center_crop = center_crop
        self.value_min = None if value_min is None else float(value_min)
        self.value_max = None if value_max is None else float(value_max)
        self.percentile_min = None if percentile_min is None else float(percentile_min)
        self.percentile_max = None if percentile_max is None else float(percentile_max)

        self.fpaths = self.get_relevant_file_paths()
        if end_idx == -1:
            self.fpaths = self.fpaths[start_idx:]
        else:
            self.fpaths = self.fpaths[start_idx:end_idx]

    def get_relevant_file_paths(self):
        extensions = ["*" + ext for ext in self.valid_extensions]
        return sorted({
            str(path) for ext in extensions
            for path in Path(self.image_root_path).rglob(ext)
        })

    def __len__(self):
        return len(self.fpaths)

    def get_shape(self):
        return (1, self.resolution, self.resolution)

    def _normalize_image(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)

        if self.value_min is not None and self.value_max is not None:
            lo = self.value_min
            hi = self.value_max
        elif self.percentile_min is not None and self.percentile_max is not None:
            lo = float(np.nanpercentile(arr, self.percentile_min))
            hi = float(np.nanpercentile(arr, self.percentile_max))
        else:
            lo = float(np.nanmin(arr))
            hi = float(np.nanmax(arr))

        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.zeros_like(arr, dtype=np.float32)

        arr = np.clip(arr, lo, hi)
        arr = (arr - lo) / (hi - lo)
        return arr.astype(np.float32)

    def _resize_and_crop(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        h, w = tensor.shape[-2:]

        scale = float(self.resolution) / min(h, w)
        new_h = max(self.resolution, int(round(h * scale)))
        new_w = max(self.resolution, int(round(w * scale)))
        tensor = F.interpolate(tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)

        if self.center_crop:
            top = max(0, (new_h - self.resolution) // 2)
            left = max(0, (new_w - self.resolution) // 2)
            tensor = tensor[:, :, top:top + self.resolution, left:left + self.resolution]
        else:
            tensor = F.interpolate(tensor, size=(self.resolution, self.resolution), mode="bilinear", align_corners=False)

        return tensor.squeeze(0)

    @staticmethod
    def _is_dicom_path(path: str) -> bool:
        return Path(path).suffix.lower() == ".dcm"

    def _load_dicom(self, fpath: str) -> np.ndarray:
        try:
            import pydicom
        except ImportError as exc:
            raise ImportError(
                "Reading .dcm files requires `pydicom`. Install it with `pip install pydicom`."
            ) from exc

        ds = pydicom.dcmread(fpath)
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr * slope + intercept
        return arr

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        if self._is_dicom_path(fpath):
            arr = self._load_dicom(fpath)
        else:
            img = Image.open(fpath)
            arr = np.asarray(img)
            if arr.ndim == 3:
                arr = arr[..., 0]
        arr = self._normalize_image(arr)
        tensor = torch.from_numpy(arr)
        tensor = self._resize_and_crop(tensor)
        return tensor * 2 - 1
