#!/bin/bash

# Specify the folder containing TIF images
image_folder="./figure01/230913"

# Specify the Python script to run
python_script="figure01.py"

# Check if the folder exists
if [ ! -d "$image_folder" ]; then
  echo "Folder not found: $image_folder"
  exit 1
fi

# Iterate through TIF images in the folder
for image_file in "$image_folder"/*.tif; do
  if [ -f "$image_file" ]; then
    echo "Processing image: $image_file"
    python "$python_script" "$image_file"
  else
    echo "No TIF images found in $image_folder"
  fi
done
