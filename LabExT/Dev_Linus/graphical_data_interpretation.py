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
    #Z = 10 ** (Z / 10)

    return X,Y,Z

def plot_data(number_of_points,first_array,second_array, tol):
    plt.style.use('_mpl-gallery')

    XL,YL,ZL = data_preparation(number_of_points, first_array)
    XR,YR,ZR = data_preparation(number_of_points, second_array)
    
    # Plot the surface
    fig, (ax_left, ax_right) = plt.subplots(1, 2, subplot_kw={"projection": "3d"}, figsize=(15, 8))

    ax_left.plot_surface(XL,YL,ZL,cmap="Oranges",alpha=0.6)
    ax_right.plot_surface(XR,YR,ZR,cmap="Oranges",alpha=0.6)

    
    ax_left.set_xlabel('x position (µm)')
    ax_left.set_ylabel('y position (µm)')
    ax_left.set_zlabel('power (dBm)')
    ax_left.set_title('Left Stage')

    ax_right.set_xlabel('x position (µm)')
    ax_right.set_ylabel('y position (µm)')
    ax_right.set_zlabel('power (dBm)')
    ax_right.set_title('Right Stage')

    ridge_and_max(XL,YL,ZL,number_of_points,ax_left,'3D', tol)
    ridge_and_max(XR,YR,ZR,number_of_points,ax_right,'3D', tol)

    fig.suptitle(f"{file_name}")

    plt.show()

def plot_data_heatmap(number_of_points, first_array, second_array, file_name, program_folder, data_showing, tol):
    plt.style.use('_mpl-gallery')

    XL,YL,ZL = data_preparation(number_of_points, first_array)
    XR,YR,ZR = data_preparation(number_of_points, second_array)
    
    # Plot the surface
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(15, 8))


    mesh_left = ax_left.pcolormesh(XL,YL,ZL,cmap="grey", shading='auto')
    mesh_right = ax_right.pcolormesh(XR,YR,ZR,cmap="grey", shading='auto')

    
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

    fig.suptitle(f"{file_name}")

    plt.tight_layout()

    ridge_and_max(XL,YL,ZL,number_of_points,ax_left,'2D', tol)
    ridge_and_max(XR,YR,ZR,number_of_points,ax_right,'2D', tol)

    if data_showing == 0:
        fig.savefig(program_folder / 'Ridge_analysis' / f'{file_name}.png', dpi=300)

    else:
        plt.show()

def ridge_and_max(X, Y, used_array, N, axes, dim, tol):
    maximum = -100
    max_coord = [X[N//2][N//2],Y[N//2][N//2],used_array[N//2][N//2]]
    offset = 0.00001
    tolerance = tol

    for i in range(1,N-1):
        for j in range(1,N-1):
            is_x_ridge = (used_array[i][j]-used_array[i][j-1]) > tolerance and (used_array[i][j]-used_array[i][j+1]) > tolerance

            is_y_ridge = (used_array[i][j]-used_array[i-1][j]) > tolerance and (used_array[i][j]-used_array[i+1][j]) > tolerance

            is_diag_ridge = ((used_array[i][j]-used_array[i-1][j-1]) > tolerance and (used_array[i][j]-used_array[i+1][j+1]) > tolerance 
                        or (used_array[i][j]-used_array[i-1][j+1]) > tolerance and (used_array[i][j]-used_array[i+1][j-1]) > tolerance)

            hits = sum([is_y_ridge, is_x_ridge, is_diag_ridge])

            if hits >= 2:
                color = "#FFA200"
            elif is_x_ridge:
                color = "#FF0000"
            elif is_y_ridge:
                color = "#23950C"
            elif is_diag_ridge:
                color = "#BC56EF"
            else:
                color = None

            
            if color != None:
                if dim=='3D':
                    axes.scatter(X[i][j],Y[i][j],used_array[i][j]+offset,c=color)
                else:
                    axes.scatter(X[i][j],Y[i][j],c=color)

    i_max, j_max = np.unravel_index(np.argmax(used_array), used_array.shape)
    max_coord = [X[i_max][j_max], Y[i_max][j_max], used_array[i_max][j_max]]

    if dim=='3D':
        axes.scatter(max_coord[0],max_coord[1],max_coord[2]+offset,c='blue')
    else:
        axes.scatter(max_coord[0],max_coord[1],c='blue')

def interpretation_script(file_name, program_folder, data_showing, tol):
    #readout data
    #file_name = '260925_measurement_1580_1.0_31_0.5'

    parameter_list = []
    left_list = []
    right_list = []

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
    if data_showing == 1:
        plot_data(N, left_array, right_array, tol)
        plot_data_heatmap(N, left_array, right_array,file_name,program_folder,data_showing, tol)
    else:
        plot_data_heatmap(N, left_array, right_array,file_name,program_folder,data_showing, tol)

def find_matching_files(folder, id_string):
    """Return the filename stems (no .csv extension) of every CSV in
    folder whose name contains id_string."""
    return [p.stem for p in folder.glob('*.csv') if id_string in p.stem]

def ridge_analysis_png_script(program_folder, id_string, tol):
    name_list = find_matching_files(program_folder,id_string)

    for name in name_list:
        interpretation_script(name, program_folder,0, tol)

program_folder = Path(__file__).resolve().parent

file_name = '260925_measurement_1570_1.0_31_0.5'
id_name = 'ID7'
ridge_tolerance = 0

ridge_analysis_png_script(program_folder, id_name, ridge_tolerance)
#interpretation_script(file_name, program_folder,1, ridge_tolerance)