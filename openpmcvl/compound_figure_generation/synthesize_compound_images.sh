#!/bin/bash

# Add SBATCH directives here

# Set up the environment
export PROJECT_ROOT=/path/to/your/project # <-- Replace with actual path
cd $PROJECT_ROOT || exit 1
export PYTHONPATH=$PROJECT_ROOT
source /path/to/venv/bin/activate # <-- Replace with actual path to virtual env

SPLIT="VALIDATION"

if [ "$SPLIT" == "TRAIN" ]; then
    INPUT_DIRS=(
        /path/to/train_radiology_images # <-- Replace with actual path
        /path/to/train_histopathology_images # <-- Replace with actual path
        /path/to/train_dermatology_images # <-- Replace with actual path
        /path/to/train_retina_images # <-- Replace with actual path
        /path/to/train_plot_images # <-- Replace with actual path
    )

    INPUT_MODALITIES=(
        "radiology"
        "histopathology"
        "dermatology"
        "retina"
        "plot"
    )
    SAVE_DIR=/path/to/save/train_compounds # <-- Replace with actual save path
    N=500000

elif [ "$SPLIT" == "VALIDATION" ]; then
    INPUT_DIRS=(
        /path/to/val_radiology_images # <-- Replace with actual path
        /path/to/val_histopathology_images
        /path/to/val_dermatology_images
        /path/to/val_retina_images
        /path/to/val_plot_images
    )

    INPUT_MODALITIES=(
        "radiology"
        "histopathology"
        "dermatology"
        "retina"
        "plot"
    )
    SAVE_DIR=/path/to/save/val_compounds # <-- Replace with actual save path
    N=20000
else
    echo "Invalid SPLIT: $SPLIT"
    exit 1
fi

echo "Input directories: ${INPUT_DIRS[*]}"
echo "Input modalities: ${INPUT_MODALITIES[*]}"
echo "Split: $SPLIT"
echo "Output directory: $SAVE_DIR"
echo "Number of images: $N"

# Run the Python script
python synthesize_compound_images.py \
    --input_dir "${INPUT_DIRS[@]}" \
    --modalities "${INPUT_MODALITIES[@]}" \
    --output_dir "$SAVE_DIR" \
    --num_images "$N"