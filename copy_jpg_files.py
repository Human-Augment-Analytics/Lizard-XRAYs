import os
import shutil
import argparse

def copy_jpg_files(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Walk through the input folder
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith('.jpg'):
                # Construct full file path
                file_path = os.path.join(root, file)
                # Copy to the output folder
                shutil.copy(file_path, output_folder)
                print(f"Copied: {file_path} -> {output_folder}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Recursively copy .jpg files from one folder to another.")
    parser.add_argument('input_folder', type=str, help="The input folder to search for .jpg files.")
    parser.add_argument('output_folder', type=str, help="The output folder to copy .jpg files to.")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Call the function with provided arguments
    copy_jpg_files(args.input_folder, args.output_folder)
