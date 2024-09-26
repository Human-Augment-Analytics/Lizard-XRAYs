import os
import shutil
import argparse

def extract_images_from_tps(tps_file_path, input_folder, output_folder):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    with open(tps_file_path, 'r') as tps_file:
        lines = tps_file.readlines()
        
    # Loop through the lines to find image entries
    for line in lines:
        if line.startswith("IMAGE="):
            image_filename = line.split("=")[1].strip()
            input_image_path = os.path.join(input_folder, image_filename)
            output_image_path = os.path.join(output_folder, image_filename)
            
            # Check if the image exists in the input folder
            if os.path.exists(input_image_path):
                # Copy the image to the output folder
                shutil.copy(input_image_path, output_image_path)
                print(f"Copied {image_filename} to {output_folder}")
            else:
                print(f"Image {image_filename} not found in {input_folder}")

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Extract images specified in a .tps file.')
    parser.add_argument('tps_file_path', type=str, help='Path to the .tps file')
    parser.add_argument('input_folder', type=str, help='Path to the input folder containing images')
    parser.add_argument('output_folder', type=str, help='Path to the output folder to save selected images')
    
    args = parser.parse_args()
    
    extract_images_from_tps(args.tps_file_path, args.input_folder, args.output_folder)

if __name__ == "__main__":
    main()
