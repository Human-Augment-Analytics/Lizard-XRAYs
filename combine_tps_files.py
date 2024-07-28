import os
import sys

def combine_tps_files(root_directory, output_file):
    # List to store paths of all .tps files
    tps_files = []

    # Walk through the directory and its subdirectories to find .tps files
    for subdir, _, files in os.walk(root_directory):
        for file in files:
            if file.lower().endswith('.tps'):
                tps_files.append(os.path.join(subdir, file))

    # Sort the files by name to ensure a consistent order
    tps_files.sort()

    # Open the output file in write mode
    with open(output_file, 'w') as outfile:
        # Iterate through each .tps file
        for tps_file in tps_files:
            # Open the .tps file in read mode
            with open(tps_file, 'r') as infile:
                # Read the contents and write to the output file
                outfile.write(infile.read())

    print(f"Combined {len(tps_files)} files into {output_file}")

if __name__ == "__main__":
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 3:
        print("Usage: python combine_tps_files.py <root_directory> <output_file>")
        sys.exit(1)

    # Get the command line arguments
    root_directory = sys.argv[1]
    output_file = sys.argv[2]

    # Call the function to combine the .tps files
    combine_tps_files(root_directory, output_file)
