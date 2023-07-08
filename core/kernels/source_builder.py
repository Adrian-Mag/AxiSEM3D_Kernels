from ..handlers.element_output import ElementOutput
from ..handlers.obspy_output import ObspyfiedOutput
from ...aux.helper_functions import window_data

import matplotlib.pyplot as plt
import numpy as np
import os

def save_STF(directory, master_time, STF, channel_type):
    for index, channel in enumerate(channel_type):
        # Save results to a text file
        filename = directory + channel + '.txt'
        # Combine time and data arrays column-wise
        combined_data = np.column_stack((master_time, STF[channel]))
        print(STF[channel])
        # Save the combined data to a text file
        np.savetxt(filename, combined_data, fmt='%.16f', delimiter='\t')


def L2_STF_builder(real_data_path: str, forward_data_path: str,
                   station: str, network: str, location:str, 
                   window_left: float=None, 
                   window_right: float=None):
    ################
    # LOAD REAL DATA
    ################
    real_data_obspyobj = ObspyfiedOutput(mseed_file_path = real_data_path)

    stream_real_data  = real_data_obspyobj.stream
    real_data_name = real_data_obspyobj.mseed_file_name
    # get real data time and time step
    real_data_time = stream_real_data[0].times('timestamp')- 5
    dt_real_data = real_data_time[1] - real_data_time[0]
    # Select the specific station data
    stream_real_data = stream_real_data.select(station=station)

    inventory = real_data_obspyobj.inv
    # Extract the station coordinates from that inventory
    inventory = inventory.select(network=network, 
                                station=station, 
                                location=location)
    station_depth = -inventory[0][0].elevation
    station_latitude = inventory[0][0].latitude
    station_longitude = inventory[0][0].longitude    

    #####################
    # LOAD SYNTHETIC DATA
    #####################
    # Get the synthetic data
    element_data_name = forward_data_path.split('/')[-1].split('.')[0]
    element_obj = ElementOutput(forward_data_path)
    sta_rad = element_obj.Earth_Radius - station_depth
    point = [sta_rad, station_latitude, station_longitude]
    stream_forward_data = element_obj.stream(point, channels=['U'])
    forward_time = element_obj.data_time
    dt_forward = forward_time[1] - forward_time[0]
    channel_type = element_obj.coordinate_frame

    #################
    # COMPUTE RESIDUE
    #################
    # Find the master time (minmax/maxmin)
    t_max = min(real_data_time[-1], forward_time[-1])
    t_min = max(real_data_time[0], forward_time[0])
    dt = max(dt_real_data, dt_forward)
    master_time = np.arange(t_min, t_max, dt)

    # Project both arrays on the master time
    interpolated_real_data = []
    interpolated_forward_data = []
    residue = {}

    fig, axs = plt.subplots(3, 1)
    for index, channel in enumerate(channel_type):
        interpolated_real_data = np.interp(master_time, 
                                        real_data_time, 
                                        stream_real_data.select(channel='U' + channel)[0].data)
        interpolated_forward_data = np.interp(master_time, 
                                            forward_time, 
                                            stream_forward_data.select(channel='U' + channel)[0].data)
        if window_left is not None and window_right is not None:
            _, windowed_real_data = window_data(master_time, interpolated_real_data, window_left, window_right)
            windowed_master_time, windowed_forward_data = window_data(master_time, interpolated_forward_data, 
                                                                  window_left, window_right)
        else:
            windowed_master_time = master_time
            windowed_real_data = interpolated_real_data
            windowed_forward_data = interpolated_forward_data
        # Apply the t -> T-t transformation to the dresidue and multiply with -1
        residue[channel] = -np.flip(np.array(windowed_forward_data) - np.array(windowed_real_data))
        axs[index].plot(windowed_master_time, residue[channel])
        axs[index].plot(windowed_master_time, windowed_forward_data, color='red')
        axs[index].plot(windowed_master_time, windowed_real_data, color='blue')
        axs[index].text(1.05, 0.5, index, transform=axs[index].transAxes)
    # Apply the t -> T-t transformation to the time 
    transformed_windowed_master_time = np.flip(np.max(master_time) - windowed_master_time)
    STF = residue

    ##########
    # PLOT STF
    ##########
    plt.show()

    ###############
    # SAVE STF FILE
    ###############
    # Save residue as STF.txt file ready to be given to AxiSEM3D
    directory = 'CMB/STFS/' + element_data_name + '_' + real_data_name + '/'
    if not os.path.exists(directory):
        ans = input('Overwrite the existing data [y/n]: ')
        if ans == 'y':
            os.makedirs(directory)
            print("Directory created:", directory)
            save_STF(directory, transformed_windowed_master_time, STF, channel_type)
    else:
        print("Directory already exists:", directory)
        ans = input('Overwrite the existing data [y/n]: ')
        if ans == 'y':
            save_STF(directory, transformed_windowed_master_time, STF, channel_type)