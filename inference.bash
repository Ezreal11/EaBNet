#!/bin/bash

# Path to the folder containing the audio files
AUDIO_FOLDER="testaudio/noisy"
OUTPUT_FOLDER="testaudio/enhanced"
# Loop through each file in the audio folder
for file in "$AUDIO_FOLDER"/*.wav; do
    # Check if the file is a regular file
    if [[ -f "$file" ]]; then
        # Run the enhance.py script on the file
        echo "Processing $file"
        filename=$(basename "$file")    # 获取文件名
        python "enhance.py" "$file" "$OUTPUT_FOLDER/$filename" 
    fi
done
