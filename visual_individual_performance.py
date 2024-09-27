import xml.etree.ElementTree as ET
import numpy as np
import argparse
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import cv2
import os

def parse_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # Dictionary to store image filenames and their corresponding landmarks
    data = {}
    
    for image in root.findall('.//image'):
        image_file = image.get('file')
        parts = {}
        for part in image.findall('.//part'):
            part_name = int(part.get('name'))
            x = float(part.get('x'))
            y = float(part.get('y'))
            parts[part_name] = (x, y)
        
        data[image_file] = parts
    return data

def find_groundtruth(file_path, image_name):
    tree = ET.parse(file_path)
    root = tree.getroot()
    ground_truth = []
    X = []
    Y = []
    for image in root.findall('.//image'):
        image_file = image.get('file')
        # print(image_file)
        if image_file == image_name:
            for part in image.findall('.//part'):
                # print(float(part.get('x')))
                x = int(part.get('x'))
                y = int(part.get('y'))
                X.append(x)
                Y.append(y)
    return [X, Y]

def find_output(file_path, image_name):
    tree = ET.parse(file_path)
    root = tree.getroot()
    ground_truth = []
    X = []
    Y = []
    for image in root.findall('.//image'):
        image_file = image.get('file')
        image_file = image_file.replace('./', '')
        if image_file == image_name:
            for part in image.findall('.//part'):
                # print(float(part.get('x')))
                x = int(part.get('x'))
                y = int(part.get('y'))
                X.append(x)
                Y.append(y)
    return [X, Y]

def get_image_name(lizard_number, groundtruth_xml):
    tree = ET.parse(groundtruth_xml)
    root = tree.getroot()
    for i, image in enumerate(root.findall('.//image')):
        # print(image.get('file'))
        # print(i, lizard_number)
        if int(i) == int(lizard_number):
            # print(image.get('file'))
            return image.get('file')

def main(groundtruth_xml, output_xml, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Parse the groundtruth XML to find the number of images
    tree = ET.parse(groundtruth_xml)
    root = tree.getroot()
    num_images = len(root.findall('.//image'))
    # for lizard_number in range(59):
    for lizard_number in range(num_images):
        name = get_image_name(lizard_number, groundtruth_xml)
        groundtruth = find_groundtruth(groundtruth_xml, name)
        

        # groundtruth = find_groundtruth(groundtruth_xml, lizard_jpeg)
        output = find_output(output_xml, name)
        image = cv2.imread(name)
        plt.imshow(image)

        plt.scatter(groundtruth[0], groundtruth[1], s = 2, color = "lawngreen", label = "Ground Truth")
        plt.scatter(output[0], output[1], s = 2, color = "deeppink", label = "Model Output", alpha=.7)
        plt.legend()
        plt.axis('off')
        plt.savefig(f"{lizard_number}_test_set.png", bbox_inches='tight', dpi=300)
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize lizard images")
    parser.add_argument("output_xml", type=str, help="Path to the output XML file")
    parser.add_argument("test_xml", type=str, help="Path to the test XML file")
    parser.add_argument("output_folder", type=str, help="Name of output folder")
    args = parser.parse_args()

    
    main(args.test_xml, args.output_xml, args.output_folder)
    