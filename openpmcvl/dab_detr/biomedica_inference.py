import os
import numpy as np
import torch
from utils import idx2lab, lab2idx
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw
from utils import build_dab_detr_model, process_detections
from time import time
from tqdm import tqdm


class LocalDataset:
    def __init__(self, image_paths, image_processor):
        self.image_paths = image_paths
        self.image_processor = image_processor
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            return self.__getitem__(idx + 1) if idx + 1 < len(self.image_paths) else None
        image_orig = image.copy()
        image_size = image.size
        image = np.array(image)
        inputs = self.image_processor(images=[image], return_tensors="pt")["pixel_values"][0]
        
        return {"image_path": image_path, "pixel_values": inputs, "image_size": image_size}

def collate_fn_local(batch):
    data = {}
    data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
    data["image_paths"] = [x["image_path"] for x in batch]
    data["image_sizes"] = [x["image_size"] for x in batch]
    return data

def split_figures(model,
                  image_processor,
                  input_img_files,
                  save_dir,
                  txt_save_path,
                  start_idx=0,
                  end_idx=None,
                  score_threshold=0.2,
                  nms_func=process_detections, 
                  nms_threshold=0.2):

    with open(input_img_files, "r") as f:
        input_image_paths = [line.strip() for line in f]

    if end_idx is None:
        end_idx = len(input_image_paths) + 1
        
    input_image_paths = input_image_paths[start_idx:end_idx]

    print(f"Processing {len(input_image_paths)} images from {start_idx} to {end_idx}...")

    # Track already processed image paths
    processed_images = set()
    if os.path.exists(txt_save_path):
        with open(txt_save_path, "r") as f:
            for line in f:
                try:
                    subfig_path, orig_path = line.strip().split("|::|")
                    processed_images.add(orig_path)
                except Exception as e:
                    print(f"Skipping malformed line: {line.strip()}")
    
    print(len(processed_images))

    # Filter out already processed images
    input_image_paths = [p for p in input_image_paths if p not in processed_images]

    print(f"Processing {len(input_image_paths)} images...")

    if not input_image_paths:
        print("All images in this split are already processed.")
        return

    dataset = LocalDataset(input_image_paths, image_processor)
    dataloader = DataLoader(dataset, batch_size=128, collate_fn=collate_fn_local, shuffle=False)

    with torch.no_grad():
        for i_b, batch in tqdm(enumerate(dataloader)):
            with open(txt_save_path, "a") as f:
                t_0 = time()
                batch["pixel_values"] = batch["pixel_values"].to(device)
                outputs = model(batch["pixel_values"], labels=None)
                target_sizes = torch.tensor([img_size[::-1] for img_size in batch["image_sizes"]])
                results = image_processor.post_process_object_detection(outputs, threshold=score_threshold, target_sizes=target_sizes)

                for image_path, image_size, result in zip(batch["image_paths"], batch["image_sizes"], results):
                    try:
                        picked_bboxes, picked_scores = nms_func(result['boxes'].cpu(), result['scores'].cpu().numpy(), nms_threshold)
                        result['boxes'] = picked_bboxes
                        result['scores'] = torch.tensor(picked_scores)
                        result['labels'] = torch.tensor([1] * len(picked_bboxes))
                    except Exception as e:
                        print(f"Error in NMS for {image_path}: {e}")

                    img_width, img_height = image_size
                    image = Image.open(image_path).convert("RGB")
                    assert image.size == (img_width, img_height), f"Image size mismatch: {image.size} != {image_size}"

                    if len(result["boxes"]) == 0:
                        print(f"No boxes detected for {image_path}")
                        subfig_path = os.path.join(save_dir, f"subfig_{0}_{os.path.basename(image_path)}")
                        image.save(subfig_path)
                        f.write(f"{subfig_path}|::|{image_path}\n")
                        continue

                    for box, score, label, i in zip(result["boxes"], result["scores"], result["labels"], range(len(result["boxes"]))):
                        x_min, y_min, x_max, y_max = [round(coord) for coord in box]
                        x_min = max(0, min(x_min, img_width - 1))
                        y_min = max(0, min(y_min, img_height - 1))
                        x_max = max(0, min(x_max, img_width - 1))
                        y_max = max(0, min(y_max, img_height - 1))

                        subfig = image.crop((x_min, y_min, x_max, y_max))
                        subfig_path = os.path.join(save_dir, f"subfig_{i}_{os.path.basename(image_path)}")
                        subfig.save(subfig_path)
                        f.write(f"{subfig_path}|::|{image_path}\n")
                
                print(f"Time taken for batch {i_b}: {time() - t_0:.2f} seconds")

if __name__ == "__main__":
    import sys

    # Modify the following variables
    checkpoint = "path/to/checkpoint"
    input_img_files = "/path/to/txt file containing image paths"
    subfig_save_dir = "/path/to/save subfigures"
    txt_save_path = f"path/to/save/subfigures/txt/file"
    #########################################################

    os.makedirs(subfig_save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, image_processor = build_dab_detr_model(checkpoint, device, idx2lab, lab2idx, image_size=480)
    model.eval()

    if not os.path.exists(input_img_files):
        print(f"File {input_img_files} does not exist.")
        sys.exit(1)
    
    start = 0
    end = None
    split_figures(model, image_processor, input_img_files, subfig_save_dir, txt_save_path, start_idx=start, end_idx=end)