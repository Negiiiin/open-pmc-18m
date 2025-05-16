"""18M Filtered Biomedica"""

import json
import os
from typing import Callable, Dict, Literal, Optional, Union

import torch
from mmlearn.conf import external_store
from mmlearn.constants import EXAMPLE_INDEX_KEY
from mmlearn.datasets.core import Modalities
from mmlearn.datasets.core.example import Example
from omegaconf import MISSING
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


@external_store(group="datasets", root_dir="dataset_root_dir")
class SubfiguresBiomedicaFiltered(Dataset[Example]):
    """PMC-OA dataset.

    Parameters
    ----------
    root_dir : str
        Path to the root folder containing jsonl file with data entries.
    split : {"train", "valid", "test"}
        Dataset split.
    include_extra: bool, default=False
        Whether or not to include the additional data samples extracted by us
        in October 2024.
    use_full_caption : bool, default=False
        Use full captions or not.
    transform : Optional[Callable], default=None
        Transform applied to images.
    tokenizer : Optional[Callable], default=None
        Function applied to textual captions.
    """

    def __init__(
        self,
        root_dir: str,
        split: Literal["train", "valid", "test"] = "train",
        mode = "Whole",
        include_extra: bool = False,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        tokenizer: Optional[
            Callable[[str], Union[torch.Tensor, Dict[str, torch.Tensor]]]
        ] = None,
    ) -> None:
        """Initialize the dataset."""

        data_path = os.path.join(root_dir, f"classified_combined_stripped.txt")
        print(f"Loading {data_path}...")
        with open(data_path, "r") as f:
            entries = [line.strip() for line in f]

        print(f"AFTER READING {data_path}...")
        print(f"--------------------------------------- {len(entries)} ------------------------------------------------------------")

        self.entries = entries

        self.root_dir = root_dir

        if transform is None:
            self.transform = ToTensor()
        else:
            self.transform = transform

        self.tokenizer = tokenizer

    def __getitem__(self, idx: int) -> Example:
        """Return the idx'th data sample."""
        subfig_path = self.entries[idx]
        subfig_path, fig_path = subfig_path.strip().split("|::|")
        subfig_path = os.path.join(self.root_dir, "subfigures_final", subfig_path)

        fig_filename = os.path.basename(fig_path)
        caption_filename = os.path.splitext(fig_filename)[0] + ".txt"
        caption_path = os.path.join(self.root_dir, "captions", caption_filename)

        try:
            with Image.open(subfig_path) as img:
                image = img.convert("RGB")
            with open(caption_path, "r") as f:
                caption = f.read().strip()
        except Exception as e:
            print(
                f"Error loading image for entry {idx}: image_path={subfig_path}",
                e,
            )
            idx = (idx + 1) % len(self.entries)
            return self.__getitem__(idx)

        if len(caption) == 0:
            print(
                f"Empty caption for entry {idx}: image_path={subfig_path}, caption={caption}"
            )
            idx = (idx + 1) % len(self.entries)
            return self.__getitem__(idx)

        if self.transform is not None:
            image = self.transform(image)

        tokens = self.tokenizer(caption) if self.tokenizer is not None else None

        example = Example(
            {
                Modalities.RGB.name: image,
                Modalities.TEXT.name: caption,
                EXAMPLE_INDEX_KEY: idx,
                Modalities.TEXT.target: 0,
                "image_path": subfig_path,
            }
        )

        if tokens is not None:
            if isinstance(tokens, dict):  # output of HFTokenizer
                assert (
                    Modalities.TEXT.name in tokens
                ), f"Missing key `{Modalities.TEXT.name}` in tokens."
                example.update(tokens)
            else:
                example[Modalities.TEXT.name] = tokens

        return example

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.entries)