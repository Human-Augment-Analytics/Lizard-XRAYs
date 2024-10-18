import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2
def read_comma_delimited_file(file_path):
    """Read a comma-delimited text file and return lists of processed and dorsal image filenames."""
    processed_images = []
    dorsal_images = []

    with open(file_path, 'r') as file:
        for line in file:
            # Strip whitespace and split by comma
            line = line.strip()
            if line:  # Ensure the line is not empty
                processed_image, dorsal_image = line.split(',')
                processed_images.append(processed_image)
                dorsal_images.append(dorsal_image)
        return processed_images, dorsal_images
def read_comma_delimited_file_2(file_path):
    """Read a comma-delimited text file and return lists of processed and dorsal image filenames."""
    image_names = []
    image_quality = []
    image_notes = []

    with open(file_path, 'r') as file:
        for line in file:
            # Strip whitespace and split by comma
            line = line.strip()
            if line:  # Ensure the line is not empty
                name, quality, note = line.split(',')
                image_names.append(name)
                image_quality.append(quality)
                image_notes.append(note)

    return image_names, image_quality, image_notes
def read_tps_file_validated(file_path, valid_images):
    """Read a TPS file and return data for entries that match valid image names."""
    data = []
    current_x = []
    current_y = []
    current_image = None
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('LM='):
                # Start reading landmarks
                reading_landmarks = True
                current_x = []  # Reset coordinates for the new entry
                current_y = []
            elif line.startswith('IMAGE='):
                current_image = line[6:]
                # print("possible image", line[6:])
                if current_image != None and current_image in valid_images:
                    # print("Current Image", current_image)
                    # index = valid_images.index(current_image)
                    data.append((current_image, current_x, current_y))
                current_image = line.split('=')[1]
                current_x = []  # Reset coordinates for the new entry
                current_y = []
                reading_landmarks = False
            elif reading_landmarks and line:  # Process point lines
                points = list(map(float, line.split()))
                current_x.append(points[0])
                current_y.append(points[1])
    
        # Add the last entry if it matches and has coordinates
        if current_image and current_image in valid_images and current_x and current_y:
            data.append((current_image, current_x, current_y))

    return data
if __name__ == "__main__":
    # updated_morph\image_mapping.txt
    txt_names = "image_mapping.txt"
    processed_images, dorsal_images = read_comma_delimited_file(txt_names)
    txt_names = "graded_lizards.txt"
    name, quality, note = read_comma_delimited_file_2(txt_names)
    value,counts = np.unique(quality, return_counts=True)
    print(value,counts)
    index = np.array(quality) == 'good'
    dorsal_names = np.array(name)[index]

    auto_names = []
    for i in range(len(dorsal_names)):
        index = dorsal_images.index(dorsal_names[i])
        auto_names.append(processed_images[index])
        # print(dorsal_names[i], auto_names[-1])
    print(auto_names[0], dorsal_names[0])
    new_dorsal_tps = read_tps_file_validated("combined_manual.tps", dorsal_names)
    print(new_dorsal_tps[0])
    with open("subset_manual.tps", "w") as file:
        for i in range(len(new_dorsal_tps)):
            lm = len(new_dorsal_tps[i][1])
            file.write(f"LM={lm}\n")
            for j in range(len(new_dorsal_tps[i][1])):
                file.write(f"{new_dorsal_tps[i][1][j]} {new_dorsal_tps[i][2][j]}\n")
            file.write(f"IMAGE={new_dorsal_tps[i][0]}\n")
    new_processed_tps = read_tps_file_validated("combined.tps", auto_names)
    with open("subset_auto.tps", "w") as file:
        for i in range(len(new_processed_tps)):
            lm = len(new_processed_tps[i][1])
            file.write(f"LM={lm}\n")
            for j in range(len(new_processed_tps[i][1])):
                file.write(f"{new_processed_tps[i][1][j]} {new_processed_tps[i][2][j]}\n")
            file.write(f"IMAGE={new_processed_tps[i][0]}\n")
    print(len(new_processed_tps))