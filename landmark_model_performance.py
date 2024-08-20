import xml.etree.ElementTree as ET
import numpy as np
import argparse
import numpy as np
from collections import defaultdict

# maps to test to output
# maps ground truth landmark number to model output landmark number
test_to_output_map = {
0: 0,
1: 1,
2: 12,
3: 23,
4: 28,
5: 29,
6: 30,
7: 31,
8: 32,
9: 33,
10: 2,
11: 3,
12: 4,
13: 5,
14: 6,
15: 7,
16: 8,
17: 9,
18: 10,
19: 11,
20: 13,
21: 14,
22: 15,
23: 16,
24: 17,
25: 18,
26: 19,
27: 20,
28: 21,
29: 22,
30: 24,
31: 25,
32: 26,
33: 27
}

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

def calculate_avg_ruler_length_pixels(testData):
    distances = []
    for image_file, test_parts in testData.items():
        zero_coords = None
        one_coords = None
        for test_part_name, test_coords in test_parts.items():
            if test_part_name == 1:
                one_coords = test_coords
            elif test_part_name == 0:
                zero_coords = test_coords
        diff_x = zero_coords[0] - one_coords[0]
        diff_y = zero_coords[1] - one_coords[1]
        diff = np.sqrt(diff_x**2 + diff_y**2)
        distances.append(diff)
    return np.mean(distances)

def calculate_differences(outputData, testData):
    differencesMap = defaultdict(list)

    for image_file, output_parts in outputData.items():
        if image_file in testData:
            test_parts = testData[image_file]
            for test_part_name, test_coords in test_parts.items():
                if test_part_name in test_to_output_map:
                    output_part_name = test_to_output_map[test_part_name]
                    if output_part_name in output_parts:
                        output_coords = output_parts[output_part_name]
                        diff_x = test_coords[0] - output_coords[0]
                        diff_y = test_coords[1] - output_coords[1]
                        diff = np.sqrt(diff_x**2 + diff_y**2)
                        differencesMap[test_part_name].append(round(diff, 2))

    return differencesMap

def main(output_xml, test_xml):
    # Parse the XML files
    output_data = parse_xml(output_xml)
    test_data = parse_xml(test_xml)

    avg_ruler_length = calculate_avg_ruler_length_pixels(test_data)
    conversion_factor = float(27 / avg_ruler_length)
    
    # Calculate the differences between corresponding landmarks
    differencesMap = calculate_differences(output_data, test_data)
    
    for landmark in differencesMap:
        avg = round(np.mean(differencesMap[landmark]) * conversion_factor, 2)
        print("Landmark = ", landmark, " Average = ", avg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate the average difference and deviation between landmarks in two XML files.")
    parser.add_argument("output_xml", type=str, help="Path to the output XML file")
    parser.add_argument("test_xml", type=str, help="Path to the test XML file")
    args = parser.parse_args()
    
    main(args.output_xml, args.test_xml)
