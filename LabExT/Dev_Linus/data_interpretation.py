import time
import csv
import re

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def data_preparation(number_of_points, used_array):
    X = used_array[:, 0].reshape(number_of_points,number_of_points)
    Y = used_array[:, 1].reshape(number_of_points,number_of_points)
    Z = used_array[:, 2].reshape(number_of_points,number_of_points)
    Z = 10 ** (Z / 10)

    return X,Y,Z
    

def plot_data(number_of_points,first_array,second_array):
    plt.style.use('_mpl-gallery')

    XL,YL,ZL = data_preparation(number_of_points, first_array)
    XR,YR,ZR = data_preparation(number_of_points, second_array)
    
    # Plot the surface
    fig, (ax_left, ax_right) = plt.subplots(1, 2, subplot_kw={"projection": "3d"}, figsize=(15, 8))

    ax_left.plot_surface(XL,YL,ZL,cmap="Oranges")
    ax_right.plot_surface(XR,YR,ZR,cmap="Oranges")

    
    ax_left.set_xlabel('x position (µm)')
    ax_left.set_ylabel('y position (µm)')
    ax_left.set_zlabel('power (dBm)')
    ax_left.set_title('Left Stage')

    ax_right.set_xlabel('x position (µm)')
    ax_right.set_ylabel('y position (µm)')
    ax_right.set_zlabel('power (dBm)')
    ax_right.set_title('Right Stage')

    fig.suptitle("Coupling Efficiency Dimensional Dependence")

    plt.show()

def plot_data_heatmap(number_of_points,first_array,second_array):
    plt.style.use('_mpl-gallery')

    XL,YL,ZL = data_preparation(number_of_points, first_array)
    XR,YR,ZR = data_preparation(number_of_points, second_array)
    
    # Plot the surface
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(15, 8))


    mesh_left = ax_left.pcolormesh(XL,YL,ZL,cmap="coolwarm", shading='auto')
    mesh_right = ax_right.pcolormesh(XR,YR,ZR,cmap="coolwarm", shading='auto')

    
    ax_left.set_xlabel('x position (µm)')
    ax_left.set_ylabel('y position (µm)')
    ax_left.set_title('Left Stage')
    ax_left.set_aspect('equal')
    fig.colorbar(mesh_left, ax=ax_left, label='power (mW)')

    ax_right.set_xlabel('x position (µm)')
    ax_right.set_ylabel('y position (µm)')
    ax_right.set_title('Right Stage')
    ax_right.set_aspect('equal')
    fig.colorbar(mesh_right, ax=ax_right, label='power (mW)')

    fig.suptitle("Coupling Efficiency Dimensional Dependence")

    plt.tight_layout()

    plt.show()

#readout data
file_name = 'measurement_1550_1.0_31_0.5'

parameter_list = []
left_list = []
right_list = []

program_folder = Path(__file__).resolve().parent
csv_file = program_folder / f'{file_name}.csv'

with open(csv_file, newline='') as csvfile:
    data = list(csv.reader(csvfile, delimiter=','))
    for i in range(len(data)):
        if i <= 1:
            parameter_list.append(data[i])
        elif data[i][0]== 'left':
            left_list.append([float(v) for v in data[i][1:]])
        else:
            right_list.append([float(v) for v in data[i][1:]])

left_array = np.asarray(left_list, dtype=np.float32)
right_array = np.asarray(right_list, dtype=np.float32)

match = re.search(r'grid dimension:(\d+)', parameter_list[0][2])
N = int(match.group(1))


#plot data
#plot_data(N, left_array, right_array)
plot_data_heatmap(N, left_array, right_array)

