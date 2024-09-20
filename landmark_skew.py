import xml.etree.ElementTree as ET
import numpy as np
import argparse
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

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

def calcuate_ruler_length(testData):
    # calculate length of each staple
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
    return distances
    
def calculate_differences(outputData, testData):
    # calculate difference between ground truth and test for each landmark
    differencesMap = defaultdict(list)
    Lengths = defaultdict(list)
    
    for image_file, output_parts in outputData.items():
        print(output_parts)
        if image_file.replace('./', '') in testData:
            test_parts = testData[image_file.replace('./', '')]
            for test_part_name, test_coords in test_parts.items():
                if test_part_name in test_to_output_map:
                    output_part_name = test_to_output_map[test_part_name]
                    if output_part_name in output_parts:
                        output_coords = output_parts[output_part_name]
                        diff_x = test_coords[0] - output_coords[0]
                        diff_y = test_coords[1] - output_coords[1]
                        length = np.sqrt(diff_x**2 + diff_y**2)
                        differencesMap[test_part_name].append([diff_x, diff_y])
                        Lengths[test_part_name].append(length)

    return differencesMap, Lengths
def plot_hist(ax, data, title, color='blue'):
    ax.hist(data, bins=30, edgecolor='black', color=color)
    ax.set_title(title)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')

def main(output_xml, test_xml, landmark):
    # Parse the XML files
    output_data = parse_xml(output_xml)
    test_data = parse_xml(test_xml)
    ruler_length = calcuate_ruler_length(test_data)
    length = []
    x_coords = []
    y_coords = []
    i = 0
    print("./" + list(test_data.keys())[0])
    for i in range(len(test_data.keys())):
        diff_x = test_data[list(test_data.keys())[i]][landmark][0] - output_data["./" + list(test_data.keys())[i]][landmark][0]
        x_coords.append(diff_x * float(27 / ruler_length[i]))
        diff_y = test_data[list(test_data.keys())[i]][landmark][1] - output_data["./" + list(test_data.keys())[i]][landmark][1]
        y_coords.append(diff_y * float(27 / ruler_length[i]))
        length.append(np.sqrt(x_coords[-1]**2 + y_coords[-1]**2))

    data = np.vstack([x_coords, y_coords])
    kde = gaussian_kde(data, bw_method='scott')
    x_grid, y_grid = np.mgrid[np.min(x_coords)-1:np.max(x_coords)+1:100j, np.min(y_coords)-1:np.max(y_coords)+1:100j]
    positions = np.vstack([x_grid.ravel(), y_grid.ravel()])
    kde_values = np.reshape(kde(positions).T, x_grid.shape)

    # Plot KDE
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    contour = ax[0].contourf(x_grid, y_grid, kde_values, levels=100, cmap='viridis')
    cbar = plt.colorbar(contour, orientation='vertical')
    cbar.set_label('Density')

    angles_radians = np.arctan2(y_coords, x_coords)
    angles = np.degrees(angles_radians)
    
    ax[0].scatter(0, 0)
    ax[0].scatter(x_coords, y_coords, s=5, color='red', alpha=0.5)
    print(length)
    ax[1].hist(length, bins = 10)
    ax_polar = plt.subplot(1, 3, 3, projection='polar')

    counts, bin_edges = np.histogram(angles)
    # print(counts, bin_edges)
    ax_polar.bar(bin_edges[:-1], counts, edgecolor='black')
    

    # Add labels and title
    ax[0].set_xlabel('Feature Centered X Coordinate (mm)')
    ax[0].set_ylabel('Feature Centered Y Coordinate (mm)')
    ax[0].set_title(f'Landmark {landmark} Aggregated Accuracy KDE')
    ax[1].set_xlabel('Binned Error (mm)')
    ax[1].set_ylabel('Frequency')
    ax[1].set_title(f'Landmark {landmark} Displacement Histogram')
    ax[2].axis('off') 
    plt.tight_layout()

    plt.show()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate the average difference and deviation between landmarks in two XML files.")
    parser.add_argument("output_xml", type=str, help="Path to the output XML file")
    parser.add_argument("test_xml", type=str, help="Path to the test XML file")
    parser.add_argument("landmark", type=int, help="Landmark to be visualized")
    args = parser.parse_args()
    
    main(args.output_xml, args.test_xml, args.landmark)
