import csv
import re

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def csv_to_arrays(file_name):
    program_folder = Path(__file__).resolve().parent.parent / "SearchForPeak" / "PeakSearcherData" 

    powers = []
    left_coords = []
    right_coords = []

    csv_file = program_folder / f'{file_name}.csv'

    with open(csv_file, newline='') as csvfile:
        data = list(csv.reader(csvfile, delimiter=','))
        for i in range(1,len(data)):
            left_coords.append((float(data[i][0]), float(data[i][1])))
            right_coords.append((float(data[i][2]), float(data[i][3])))
            powers.append(float(data[i][4]))

    return left_coords, right_coords, powers

def plot_data(file_name):
    left_coords, right_coords, powers = csv_to_arrays(file_name)

    fig = plt.figure(figsize=(15, 8))

    ax_left, ax_right = fig.add_subplot(1, 2, 1, projection="3d"), fig.add_subplot(1, 2, 2, projection="3d")

    x_left = [row[0] for row in left_coords]
    y_left = [row[1] for row in left_coords]

    x_right = [row[0] for row in right_coords]
    y_right = [row[1] for row in right_coords]

    sc = ax_left.scatter(x_left, y_left, powers, c=powers, cmap='viridis', s=50)
    sc = ax_right.scatter(x_right, y_right, powers, c=powers, cmap='viridis', s=50)

    ax_left.set_xlabel('x position (µm)')
    ax_left.set_ylabel('y position (µm)')
    ax_left.set_zlabel('power (mW)')
    ax_left.set_title('Left Stage')

    ax_right.set_xlabel('x position (µm)')
    ax_right.set_ylabel('y position (µm)')
    ax_right.set_zlabel('power (mW)')
    ax_right.set_title('Right Stage')

    fig.colorbar(sc, ax=ax_left, label='Z Value', pad=0.1)
    fig.colorbar(sc, ax=ax_right, label='Z Value', pad=0.1)

    plt.show()

file_name = '261125_133152_coordinate search_1580_200um_detailed_ID117'

plot_data(file_name)
