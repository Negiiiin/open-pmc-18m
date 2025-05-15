from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import json
import torch
from utils import lab2idx
from time import sleep

class SyntheticData(Dataset):
    def __init__(self, data_root, image_processor):
        self.data_root = data_root
        images_file_names = os.path.join(data_root, "generated_images.txt")
        with open(images_file_names, "r") as f:
            self.images = f.read().splitlines()
        
        self.image_processor = image_processor

    @staticmethod
    def format_image_annotations_as_coco(image_id, categories, boxes):
        annotations = []
        for category, bbox in zip(categories, boxes):
            formatted_annotation = {
                "image_id": image_id,
                "category_id": category,
                "bbox": list(bbox),
                "iscrowd": 0,
                "area": bbox[2] * bbox[3],
            }
            annotations.append(formatted_annotation)

        return {
            "image_id": image_id,
            "annotations": annotations,
        }

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx, fail_counter=0):
        image_path = os.path.join(self.data_root, self.images[idx])
        try:
            image = Image.open(image_path)
        except Exception as e:
            print("Error opening image:", e)
            print("Image path:", image_path)
            fail_counter += 1
            if fail_counter < 5:
                sleep(5)
                return self.__getitem__(idx, fail_counter)
            else:
                raise e
        image = np.array(image.convert("RGB"))

        annotations_path = image_path.replace(".jpg", ".json")
        try:
            with open(annotations_path, "r") as f:
                annot = json.load(f)
        except Exception as e:
            print("Error opening annotations:", e)
            print("Annotations path:", annotations_path)
            fail_counter += 1
            if fail_counter < 5:
                sleep(5)
                return self.__getitem__(idx, fail_counter)
            else:
                raise e
        
        boxes = annot['bboxes_coco']
        # Setting all labels to 1
        categories =[1 for _ in annot['labels']]
        formatted_annotations = self.format_image_annotations_as_coco(idx, categories, boxes)
        result = self.image_processor(
            images=image, annotations=formatted_annotations, return_tensors="pt"
        )
        result = {k: v[0] for k, v in result.items()}
        return result


def collate_fn(batch):
    data = {}
    data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
    data["labels"] = [x["labels"] for x in batch]
    return data



class BiomedicaSplittingDataset(Dataset):
    def __init__(self, images_file_names, image_processor):

        self.images_file_names = images_file_names
        
        self.image_processor = image_processor

    def __len__(self):
        return len(self.images_file_names)

    def __getitem__(self, idx):
        image_path = self.images_file_names[idx]
        try:
            image = Image.open(image_path)
        except Exception as e:
            print("Error opening image:", e)
            print("Image path:", image_path)
            return self.__getitem__((idx + 1) % len(self.images_file_names))
        image = np.array(image.convert("RGB"))
        
        result = self.image_processor(
            images=[image], return_tensors="pt"
        )
        result = {k: v[0] for k, v in result.items()}
        result["labels"] = None
        return result