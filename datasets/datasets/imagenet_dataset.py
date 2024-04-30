from typing import Tuple, List, Dict, Any, Optional
import torchvision
from .base_dataset import BaseDataset
from utils.input_checks import check_read_dir


class ImageNetDataset(BaseDataset):

    SPLIT_OPTIONS: List[str] = ['train', 'va']

    def __init__(
        self,
        data_root: str,
        split: str,
        transforms: Optional[dict] = None,
        indices: Optional[List[int]] = None,
    ) -> None:
        self.data_root = check_read_dir(data_root)
        assert type(split) == str, f"{type(split)=}"
        self.indices = indices
        self.dataset = torchvision.datasets.ImageNet(
            root=data_root, split=split,
        )
        # initialize transform
        self._init_transform_(transforms=transforms)

    def __len__(self) -> int:
        return len(self.dataset)

    def _load_example_(self, idx: int) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        if self.indices is not None:
            idx = self.indices[idx]
        image, target = self.dataset[idx]
        inputs = {'image': image}
        labels = {'target': target}
        meta_info = {
            'image_filepath': self.dataset.samples[idx][0],
            'image_resolution': tuple(image.shape[-2:]),
        }
        return inputs, labels, meta_info
