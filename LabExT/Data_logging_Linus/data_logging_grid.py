
import time
import csv

import numpy as np
from pathlib import Path


from LabExT.Movement.Stages.Stage6DSmarActMCS2 import Stage6DSmarActMCS2

from LabExT.Instruments.LaserMainframeKeysight import LaserMainframeKeysight
from LabExT.Instruments.PowerMeterN7744A import PowerMeterN7744A


#variables
#laser variables
wavelength = 1530 #nm
power = 1.0 #dBm

#data gathering variables
step_size = 10 #1 micro meter
number_of_data_points = 3 #20x20 micro meter

#define stages and instruments

addresses = Stage6DSmarActMCS2.find_stage_addresses()

print("Detected stages:")
for address in addresses:
    print(address)
    
left_stage = Stage6DSmarActMCS2(addresses[0])
right_stage = Stage6DSmarActMCS2(addresses[1])

laser = LaserMainframeKeysight(visa_address="TCPIP::100.65.34.13::INSTR", channel = 0)
powermeter = PowerMeterN7744A(visa_address="TCPIP0::100.65.33.173::INSTR", channel = 1)

#empty matrix for data collection
left_list = []
right_list = []



try:
    #setup stages
    left_check = left_stage.connect()
    right_check = right_stage.connect()

    if not (left_check and right_check):
        raise RuntimeError('Stage could not be connected.') 
    print('Stages connected.')

    #setup laser
    laser.open()
    laser.wavelength = wavelength
    laser.power = power
    laser.unit = 'dBm'
    laser.enable = True


    #setup powermeter
    powermeter.open()
    powermeter.wavelength = wavelength
    powermeter.unit = 'dbm'
    powermeter.averagetime = 0.1

    #data gathering definitions
    left_abs_max_pos=left_stage.get_position()
    left_stage.move_absolute(left_abs_max_pos[0]-(number_of_data_points-1)/2 * step_size, left_abs_max_pos[1]-(number_of_data_points-1)/2 * step_size, left_abs_max_pos[2])
    left_abs_pos=left_stage.get_position()

    for dx in range(number_of_data_points):
        for dy in range(number_of_data_points):
            left_stage.move_absolute(left_abs_pos[0]+dx*step_size,left_abs_pos[1]+dy*step_size,left_abs_pos[2])
            time.sleep(0.01)
            power_measured = powermeter.power
            x, y, z = left_stage.get_position()
            left_list.append(('left',x,y,power_measured))


    left_stage.move_absolute(left_abs_max_pos[0],left_abs_max_pos[1],left_abs_max_pos[2])

    right_abs_max_pos=right_stage.get_position()
    right_stage.move_absolute(right_abs_max_pos[0]-(number_of_data_points-1)/2 * step_size, right_abs_max_pos[1]-(number_of_data_points-1)/2 * step_size, right_abs_max_pos[2])
    right_abs_pos=right_stage.get_position()

    for dx in range(number_of_data_points):
        for dy in range(number_of_data_points):
            right_stage.move_absolute(right_abs_pos[0]+dx*step_size,right_abs_pos[1]+dy*step_size,right_abs_pos[2])
            time.sleep(0.01)
            power_measured = powermeter.power
            x, y, z = right_stage.get_position()
            right_list.append(('right',x,y,power_measured))

    right_stage.move_absolute(right_abs_max_pos[0],right_abs_max_pos[1],right_abs_max_pos[2])

finally:
    for cleanup_step in (
        lambda: setattr(laser, 'enable', False),
        laser.close,
        powermeter.close,
        left_stage.disconnect,
        right_stage.disconnect,
    ):
        try:
            cleanup_step()
        except Exception as e:
            print(f'Cleanup step failed: {e}')

#data saving
left_data = np.array(left_list)
right_data = np.array(right_list)

program_folder = Path(__file__).resolve().parent
csv_file = program_folder / "measurements.csv"

with open(csv_file, 'w',newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in left_data:
        writer.writerow(row)
    for row in right_data:
        writer.writerow(row)

