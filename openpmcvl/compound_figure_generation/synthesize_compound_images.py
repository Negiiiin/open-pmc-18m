import os
import random
import json
import argparse
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from time import time
import uuid
from multiprocessing import Pool, cpu_count


# -----------------------------------------------------------------------------
def random_gap(margin_mode, max_gap=50):
    """
    Choose a random gap using a Beta distribution, which is supported between 0 and max_gap
    and concentrates around zero.
    """
    if margin_mode == 'no margin':
        return 0
    gap = np.random.beta(a=2, b=10) * max_gap  # Beta(2,10) skews towards 0
    return int(gap)

# -----------------------------------------------------------------------------
def measure_text(draw, text, font):
    """
    Measure the size (width, height) of the given text using the provided draw context and font.
    """
    try:
        return draw.textsize(text, font=font)
    except AttributeError:
        bbox = draw.textbbox((0, 0), text, font=font)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return (width, height)

def get_contrasting_text_color(img, position, text_size, threshold=128):
    """
    Given an image, a starting position (x, y), and text_size (width, height),
    compute the average brightness in that region.
    """
    x, y = position
    region = img.crop((x, y, x + text_size[0], y + text_size[1]))
    gray = region.convert("L")
    np_gray = np.array(gray)
    avg_brightness = np_gray.mean()
    return "white" if avg_brightness < threshold else "black"

# -----------------------------------------------------------------------------
# Add label to image 
def add_label_to_image(img, label, layout_params, target_size=None):
    font = layout_params['font']
    scheme = layout_params['label_placement']
    if font is None:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(img)
    text_size = measure_text(draw, label, font)

    if scheme == 'overlay_top':
        # 🔀 Add randomness to top-left position
        x_offset = random.randint(5, 20)  # 🔸 randomness here
        y_offset = random.randint(5, 15)  # 🔸 and here
        text_color = get_contrasting_text_color(img, (x_offset, y_offset), text_size)
        draw.text((x_offset, y_offset), label, fill=text_color, font=font)
        out_img = img
        if target_size is not None and out_img.size != target_size:
            new_img = Image.new("RGB", target_size, "white")
            new_img.paste(out_img, (0, 0))
            out_img = new_img
        return out_img, (0, 0)

    elif scheme == 'overlay_bottom':
        # 🔀 Randomize left margin and slightly vary vertical offset
        x_offset = random.randint(5, 20)  # 🔸 randomness here
        y_offset = random.randint(5, 15)
        text_y = img.height - text_size[1] - y_offset  # 🔸 and here
        text_color = get_contrasting_text_color(img, (x_offset, text_y), text_size)
        draw.text((x_offset, text_y), label, fill=text_color, font=font)
        out_img = img
        if target_size is not None and out_img.size != target_size:
            new_img = Image.new("RGB", target_size, "white")
            new_img.paste(out_img, (0, 0))
            out_img = new_img
        return out_img, (0, 0)

    elif scheme == 'pad_left':
        padding = text_size[0] + 10
        new_width = img.width + padding
        new_img = Image.new("RGB", (new_width, img.height), "white")
        new_img.paste(img, (padding, 0))
        draw_new = ImageDraw.Draw(new_img)
        # 🔀 Add slight randomness to vertical alignment
        text_y = (img.height - text_size[1]) // 2 + random.randint(-5, 5)  # 🔸 here
        draw_new.text((5, text_y), label, fill="black", font=font)
        if target_size is not None and new_img.size[0] < target_size[0]:
            final_img = Image.new("RGB", target_size, "white")
            final_img.paste(new_img, (target_size[0] - new_img.size[0], 0))
            new_img = final_img
        return new_img, (padding, 0)

    elif scheme == 'pad_bottom':
        padding = text_size[1] + 10
        new_height = img.height + padding
        new_img = Image.new("RGB", (img.width, new_height), "white")
        new_img.paste(img, (0, 0))
        draw_new = ImageDraw.Draw(new_img)
        # 🔀 Horizontal alignment with a small shift
        text_x = (img.width - text_size[0]) // 2 + random.randint(-10, 10)  # 🔸 here
        draw_new.text((text_x, img.height + 5), label, fill="black", font=font)
        if target_size is not None and new_img.size[1] < target_size[1]:
            final_img = Image.new("RGB", target_size, "white")
            final_img.paste(new_img, (0, target_size[1] - new_img.size[1]))
            new_img = final_img
        return new_img, (0, padding)

    elif scheme == 'pad_above':
        padding = text_size[1] + 10
        new_height = img.height + padding
        new_img = Image.new("RGB", (img.width, new_height), "white")
        new_img.paste(img, (0, padding))
        draw_new = ImageDraw.Draw(new_img)
        # 🔀 Horizontal center with slight jitter
        text_x = (img.width - text_size[0]) // 2 + random.randint(-10, 10)  # 🔸 here
        draw_new.text((text_x, 5), label, fill="black", font=font)
        if target_size is not None and new_img.size[1] < target_size[1]:
            final_img = Image.new("RGB", target_size, "white")
            final_img.paste(new_img, (0, 0))
            new_img = final_img
        return new_img, (0, padding)

    else:
        draw.text((5, 5), label, fill="black", font=font)
        return img, (0, 0)

# -----------------------------------------------------------------------------
def generate_label(idx, layout_params):
    label_content = layout_params['label_content']
    letter_type = layout_params['letter_type']
    if not layout_params['is_special']:
        grid_dims = layout_params['grid_dims']
    else:
        grid_dims = None

    if letter_type == 'number':
        base = lambda x: str(x)
    elif letter_type == 'roman_number':
        roman_numerals = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x',
        'xi', 'xii', 'xiii', 'xiv', 'xv', 'xvi', 'xvii', 'xviii', 'xix',
        'xx', 'xxi', 'xxii', 'xxiii', 'xxiv', 'xxv', 'xxvi', 'xxvii', 'xxviii', 'xxix', 'xxx']
        base = lambda x: roman_numerals[x]
    elif letter_type == 'lower':
        base = lambda x: chr(97 + x)
    else:
        base = lambda x: chr(65 + x)
        
    if label_content == 'simple':
        return base(idx)
    elif label_content == 'paren':
        return f"({base(idx)})"
    elif label_content == 'single_paren':
        return f"{base(idx)})"
    elif label_content == 'grid':
        if grid_dims is None:
            return base(idx)
        else:
            rows, cols = grid_dims
            row = idx // cols
            col = idx % cols
            return f"{base(row)}-{col+1}"
    elif label_content == 'grid_no_dash':
        if grid_dims is None:
            return base(idx)
        else:
            rows, cols = grid_dims
            row = idx // cols
            col = idx % cols
            return f"{base(row)}{col+1}"
    elif label_content == 'grid_paren':
        if grid_dims is None:
            return f"({base(idx)})"
        else:
            rows, cols = grid_dims
            row = idx // cols
            col = idx % cols
            return f"({base(row)}-{col+1})"
    elif label_content == 'grid_single_paren':
        if grid_dims is None:
            return f"{base(idx)})"
        else:
            rows, cols = grid_dims
            row = idx // cols
            col = idx % cols
            return f"{base(row)}-{col+1})"
    elif label_content == 'grid_paren_no_dash':
        if grid_dims is None:
            return f"({base(idx)})"
        else:
            rows, cols = grid_dims
            row = idx // cols
            col = idx % cols
            return f"({base(row)}{col+1})"
    elif label_content == 'grid_single_paren_no_dash':
        if grid_dims is None:
            return f"{base(idx)})"
        else:
            rows, cols = grid_dims
            row = idx // cols
            col = idx % cols
            return f"{base(row)}{col+1})"
    else:
        return base(idx)

#------------------------------------------------------------------------------
# Choosing the layout params:
def choose_layout_params(sampling_methods,
                         sampling_weights,
                         base_subfig_size=(100, 100),
                         max_grid_size=5,
                         special_layout_prob=0.1,
                         possible_grids=None):
    if possible_grids is None:
        possible_grids = [(i, j) for i in range(1, max_grid_size + 1) for j in range(1, max_grid_size + 1)]

    aspect_ratio = random.choice([(1, 1. + i / 10.) for i in range(6)] + [(1. + i / 10., 1) for i in range(6)])
    base_size = (int(base_subfig_size[0] * aspect_ratio[0]), int(base_subfig_size[1] * aspect_ratio[1]))

    is_special = random.random() <= special_layout_prob
    extra_subfigures = 0

    letter_type = random.choices(['lower', 'upper', 'number', 'roman_number'], weights=[0.4, 0.4, 0.1, 0.1], k=1)[0]

    if is_special:
        layout_type = random.choice(['vertical_extra', 'horizontal_extra', 'row_mismatch', 'col_mismatch'])
        grid_dims = random.choice([(i, i) for i in range(2, 5)])
        extra_subfigures = 1
        label_content = random.choice(['simple', 'paren'])
    else:
        layout_type = 'standard'
        grid_dims = random.choice(possible_grids)
        if letter_type in ["number", "roman_number"]:
            label_content = random.choice(['simple', 'paren', 'single_paren'])
        else:
            label_content = random.choice(['simple', 'paren', 'grid', 'grid_no_dash', 'grid_paren', 'grid_single_paren', 'grid_paren_no_dash', 'grid_single_paren_no_dash'])

    label_placement = random.choices(['overlay_top', 'overlay_bottom', 'pad_left', 'pad_bottom', 'pad_above'], weights=[0.15, 0.15, 0.3, 0.3, 0.1], k=1)[0]

    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except Exception:
        font = ImageFont.load_default()

    modality_mode = random.choices(sampling_methods, weights=sampling_weights, k=1)[0]

    r = random.random()
    if r < 0.2:
        margin_mode = 'no margin'
    else:
        margin_mode = 'random'

    params = {'is_special': is_special,
              'layout_type': layout_type,
              'grid_dims': grid_dims,
              'extra_subfigures': extra_subfigures,
              'base_subfig_size': base_size,
              'aspect_ratio': aspect_ratio,
              'label_content': label_content,
              'label_placement': label_placement,
              'letter_type': letter_type,
              'font': font,
              'modality_mode': modality_mode,
              'margin_mode': margin_mode,
              }

    return params    

# -----------------------------------------------------------------------------
# Get cropped subfigure
def get_cropped_scaled_subfigure(img, layout_params, special_subfig=False):
    h_base, w_base = layout_params['base_subfig_size']
    h_ratio, w_ratio = layout_params['aspect_ratio']
    if special_subfig:
        if layout_params['layout_type'] == 'row_mismatch':
            w_base = layout_params['grid_dims'][1] * w_base
            w_ratio = layout_params['grid_dims'][1] * w_ratio
        elif layout_params['layout_type'] == 'col_mismatch':
            h_base = layout_params['grid_dims'][0] * h_base
            h_ratio = layout_params['grid_dims'][0] * h_ratio

    img_w, img_h = img.size
    img_mid_w, img_mid_h = img_w // 2, img_h // 2
    if h_ratio > w_ratio:
        h_orig = img_h
        w_orig = int(1.0 * img_h * w_ratio / h_ratio)
        w_orig = min(w_orig, img_w)
        x1 = img_mid_w - w_orig // 2
        x2 = img_mid_w + w_orig // 2
        y1 = 0
        y2 = img_h
    else:
        w_orig = img_w
        h_orig = int(1.0 * img_w * h_ratio / w_ratio)
        h_orig = min(h_orig, img_h)
        x1 = 0
        x2 = img_w
        y1 = img_mid_h - h_orig // 2
        y2 = img_mid_h + h_orig // 2
    return img.crop((x1, y1, x2, y2)).resize((w_base, h_base))

# -----------------------------------------------------------------------------
# Alternative to get_cropped_scaled_subfigure for plot modality.
def get_plot_subfigure(img, layout_params, special_subfig=False):
    h_base, w_base = layout_params['base_subfig_size']
    h_ratio, w_ratio = layout_params['aspect_ratio']
    if special_subfig:
        if layout_params['layout_type'] == 'row_mismatch':
            w_base = layout_params['grid_dims'][1] * w_base
            w_ratio = layout_params['grid_dims'][1] * w_ratio
        elif layout_params['layout_type'] == 'col_mismatch':
            h_base = layout_params['grid_dims'][0] * h_base
            h_ratio = layout_params['grid_dims'][0] * h_ratio

    img_w, img_h = img.size
    scale = min(w_base / img_w, h_base / img_h)
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)
    img_padded = Image.new("RGB", (w_base, h_base), "white")
    x_offset = (w_base - new_w) // 2
    y_offset = (h_base - new_h) // 2
    img_padded.paste(img_resized, (x_offset, y_offset))
    return img_padded
    
# -----------------------------------------------------------------------------
# Sample subfigures
def sample_subfigure_path(num_subfigures,
                          input_directories,
                          image_files,
                          sampling_methods,
                          layout_params):
    
    subfig_paths = []

    modality = layout_params['modality_mode']
    mod2idx = {mod: idx for idx, mod in enumerate(sampling_methods)}
    mod_idx = mod2idx[modality]
    
    if mod_idx >= len(input_directories):
        # mixed modality
        sampled_mod_indices = random.choices(range(len(input_directories)), k=num_subfigures)
        sampled_mods = [sampling_methods[idx] for idx in sampled_mod_indices]
        subfig_paths = []
        for idx in sampled_mod_indices:
            input_directory = input_directories[idx]
            image_files_mod = image_files[idx]
            subfig_paths.append(os.path.join(input_directory, random.choice(image_files_mod)))
    else:
        # single modality
        sampled_mods = [modality] * num_subfigures
        input_directory = input_directories[mod_idx]
        image_files_mod = image_files[mod_idx]
        subfig_paths = [os.path.join(input_directory, random.choice(image_files_mod)) for _ in range(num_subfigures)]
    
    return subfig_paths, sampled_mods, modality

# -----------------------------------------------------------------------------
# Get subfigures and their labels.
def get_subfigures_with_labels(input_directories, image_files, sampling_methods, layout_params):
    num_subfigures_standard = layout_params['grid_dims'][0] * layout_params['grid_dims'][1]
    num_subfigures = num_subfigures_standard + layout_params['extra_subfigures']
    labels = [generate_label(idx, layout_params) for idx in range(num_subfigures)]
    subfig_paths, sampled_mods, sample_method = sample_subfigure_path(num_subfigures, input_directories, image_files, sampling_methods, layout_params)

    final_subfigures = {"standard": [], "special": []}
    for idx in range(num_subfigures):
        is_specail = (idx >= num_subfigures_standard) and layout_params['is_special']
        subfig_path = subfig_paths[idx]
        image = Image.open(subfig_path).convert("RGB")
        modality = sampled_mods[idx]
        if modality == 'plot':
            cropped_image = get_plot_subfigure(image, layout_params, is_specail)
        else:
            cropped_image = get_cropped_scaled_subfigure(image, layout_params, is_specail)
        label = labels[idx]
        labeled_image, _ = add_label_to_image(cropped_image, label, layout_params)
        if is_specail:
            final_subfigures["special"].append((
                labeled_image,
                label,
                {"original_path": subfig_path, "full_image_size": image.size, "size": cropped_image.size, "modality": sampled_mods[idx]}
            ))
        else:
            final_subfigures["standard"].append((
                labeled_image,
                label,
                {"original_path": subfig_path, "full_image_size": image.size, "size": cropped_image.size, "modality": sampled_mods[idx]}
            ))
    
    return final_subfigures, sample_method

# -----------------------------------------------------------------------------
# Get final layout and compound image.
def get_final_layout(layout_params, subfigures, gap_func):
    special_subfigures = subfigures["special"]
    standard_subfigures = subfigures["standard"]
    # Get padding values.
    pad_left = random.choice(range(1, 20))
    pad_right = random.choice(range(1, 20))
    pad_bottom = random.choice(range(1, 20))
    pad_top = random.choice(range(1, 20))
    ##########################################################
    # outputs
    meta_data = []
    total_width = pad_left + pad_right
    total_height = pad_top + pad_bottom
    subfigures_ordered = []
    labels = []
    bboxes_coco = []
    ##########################################################
    # Get standard subfigures sizes and gaps.
    standard_heights = [subfig[0].size[1] for subfig in standard_subfigures]
    standard_widths = [subfig[0].size[0] for subfig in standard_subfigures]
    x_gaps_standards = [gap_func(layout_params['margin_mode']) for _ in range(layout_params['grid_dims'][1] - 1)]
    y_gaps_standards = [gap_func(layout_params['margin_mode']) for _ in range(layout_params['grid_dims'][0] - 1)]
    rows, cols = layout_params['grid_dims']
    standard_total_height = max(standard_heights) * rows + sum(y_gaps_standards)
    standard_total_width = max(standard_widths) * cols + sum(x_gaps_standards)
    ##########################################################
    # Special subfigures + total width and height
    x_gaps_extra = []
    y_gaps_extra = []
    below_special, above_special, right_special, left_special = False, False, False, False
    # layout_type = random.choice(['vertical_extra', 'horizontal_extra', 'row_mismatch', 'col_mismatch'])
    if layout_params['is_special']:
        if layout_params['layout_type'] in ['vertical_extra', 'col_mismatch']:
            x_gaps_extra = [gap_func(layout_params['margin_mode']) for _ in range(1)]
            left_special = random.choice([True, False])
            right_special = not left_special
            total_height += max(standard_total_height, special_subfigures[0][0].size[1])
            total_width += special_subfigures[0][0].size[0] + sum(x_gaps_extra) + standard_total_width

        elif layout_params['layout_type'] in ['horizontal_extra', 'row_mismatch']:
            y_gaps_extra = [gap_func(layout_params['margin_mode']) for _ in range(1)]
            above_special = random.choice([True, False])
            below_special = not above_special
            total_width += max(standard_total_width, special_subfigures[0][0].size[0])
            total_height += special_subfigures[0][0].size[1] + sum(y_gaps_extra) + standard_total_height
    else:
        total_height += standard_total_height
        total_width += standard_total_width

    ##########################################################
    # Get coordinates for each subfigure.
    start_x = pad_left
    start_y = pad_top
    if left_special:
        labeled_image, label, meta_data_subfigure = special_subfigures[0]
        x, y, w, h = start_x, start_y, labeled_image.size[0], labeled_image.size[1]
        bboxes_coco.append([x, y, w, h])
        labels.append(label)
        subfigures_ordered.append(labeled_image)
        meta_data.append(meta_data_subfigure)
        start_x += w + x_gaps_extra[0]
    
    if above_special:
        labeled_image, label, meta_data_subfigure = special_subfigures[0]
        x, y, w, h = start_x, start_y, labeled_image.size[0], labeled_image.size[1]
        bboxes_coco.append([x, y, w, h])
        labels.append(label)
        subfigures_ordered.append(labeled_image)
        meta_data.append(meta_data_subfigure)
        start_y += h + y_gaps_extra[0]
    
    rows, cols = layout_params['grid_dims']
    start_x_copy, start_y_copy = start_x, start_y
    for i in range(len(standard_subfigures)):
        labeled_image, label, meta_data_subfigure = standard_subfigures[i]
        labels.append(label)
        subfigures_ordered.append(labeled_image)
        meta_data.append(meta_data_subfigure)
        row, col = i // cols, i % cols
        x, y, w, h = start_x, start_y, labeled_image.size[0], labeled_image.size[1]
        bboxes_coco.append([x, y, w, h])
        if col == cols - 1 and row < rows - 1:
            start_x = start_x_copy
            start_y += h + y_gaps_standards[row]
        elif col == cols - 1 and row == rows - 1:
            start_y += h
            start_x += w
        elif col < cols - 1:
            start_x += w + x_gaps_standards[col]

    if right_special:
        start_y = start_y_copy
        start_x += x_gaps_extra[0]
        labeled_image, label, meta_data_subfigure = special_subfigures[0]
        x, y, w, h = start_x, start_y, labeled_image.size[0], labeled_image.size[1]
        bboxes_coco.append([x, y, w, h])
        labels.append(label)
        subfigures_ordered.append(labeled_image)
        meta_data.append(meta_data_subfigure)
    
    if below_special:
        start_x = start_x_copy
        start_y += y_gaps_extra[0]
        labeled_image, label, meta_data_subfigure = special_subfigures[0]
        x, y, w, h = start_x, start_y, labeled_image.size[0], labeled_image.size[1]
        bboxes_coco.append([x, y, w, h])
        labels.append(label)
        subfigures_ordered.append(labeled_image)
        meta_data.append(meta_data_subfigure)
    
    return subfigures_ordered, total_width, total_height, labels, bboxes_coco, meta_data

def generate_single_image(args):
    idx, input_dirs, image_files, sampling_methods, sampling_weights, output_dir = args

    layout_params = choose_layout_params(sampling_methods, sampling_weights)
    subfigures, sample_method = get_subfigures_with_labels(input_dirs, image_files, sampling_methods, layout_params)
    subfigures_ordered, compound_w, compound_h, labels, bboxes_coco, meta_data = get_final_layout(layout_params, subfigures, random_gap)

    compound_image = Image.new("RGB", (compound_w, compound_h), "white")
    bboxes_yolo = []
    for idx2 in range(len(labels)):
        bbox_x, bbox_y, bbox_w, bbox_h = bboxes_coco[idx2]
        sub_img = subfigures_ordered[idx2]
        compound_image.paste(sub_img, (bbox_x, bbox_y))
        cx = (bbox_x + bbox_w / 2) / compound_w
        cy = (bbox_y + bbox_h / 2) / compound_h
        norm_w = bbox_w / compound_w
        norm_h = bbox_h / compound_h
        bboxes_yolo.append([cx, cy, norm_w, norm_h])

    unique_name = f"{uuid.uuid4().hex}"
    compound_filename = f"compound_{unique_name}.jpg"
    compound_path = os.path.join(output_dir, compound_filename)
    compound_image.save(compound_path)

    data = {
        "name": compound_filename,
        "image_width": compound_w,
        "image_height": compound_h,
        "number_of_subfigures": len(labels),
        "bboxes_yolo": bboxes_yolo,
        "bboxes_coco": bboxes_coco,
        "labels": labels,
        "sampling_method": sample_method,
        "metadata": meta_data,
    }

    json_filename = f"compound_{unique_name}.json"
    json_path = os.path.join(output_dir, json_filename)
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)

    return compound_filename


def generate_compound_images(input_dirs, input_modalities, output_dir, N, num_workers=None):
    assert len(input_dirs) == len(input_modalities), "Number of input directories and modalities must match."
    os.makedirs(output_dir, exist_ok=True)

    valid_ext = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for input_dir in input_dirs:
        image_files.append([f for f in os.listdir(input_dir)
                            if os.path.splitext(f)[1].lower() in valid_ext and
                            os.path.isfile(os.path.join(input_dir, f))])
        assert len(image_files[-1]) > 0, f"No valid images found in {input_dir}"

    sampling_methods = input_modalities + ['mixed']
    sampling_weights = [1 / (len(input_dirs) + 1) for _ in range(len(input_dirs) + 1)]

    args_list = [
        (i, input_dirs, image_files, sampling_methods, sampling_weights, output_dir)
        for i in range(N)
    ]

    t = time()
    with Pool(processes=num_workers or cpu_count()) as pool:
        generated_image_names = list(pool.map(generate_single_image, args_list))

    txt_path = os.path.join(output_dir, "generated_images.txt")
    with open(txt_path, "w") as f:
        for name in generated_image_names:
            f.write(name + "\n")

    elapsed = time() - t
    print(f"Generated {N} compound images in {output_dir} in {elapsed:.2f} seconds.")

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate random compound images from subfigures.")
    parser.add_argument("--input_dir", type=str, nargs='+', required=True,
                        help="Directories containing subfigure images."
    )
    parser.add_argument("--modalities", type=str, nargs='+', required=True,
                        help="Modalities corresponding to input directories."
    )
    parser.add_argument("--output_dir", type=str, default="generated_data",
                        help="Directory to save compound images and metadata.")
    parser.add_argument("--num_images", type=int, default=10,
                        help="Number of compound images to generate (N).")
    args = parser.parse_args()

    generate_compound_images(args.input_dir, args.modalities, args.output_dir, args.num_images)