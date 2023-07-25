from .source_builder import L2_STF_builder
from ..handlers.obspy_output import ObspyfiedOutput
from ...aux.helper_functions import find_phase_window
from .kernel import L2Kernel

import numpy as np

#@@@@@@@@@@@@@
# L2 Kernels @
#@@@@@@@@@@@@@

###################
# CHOICE PARAMETERS
###################

SOURCE_BUILDING = False
KERNEL_COMPUTATION = True

############################
# SOURCE BUILDING PARAMETERS
############################
# Path to the mseed file of real data
real_data_path = '/home/adrian/PhD/AxiSEM3D/CMB/kernels/real/output/stations/Station_grid/obspyfied/Station_grid.mseed'
# Path to element output of synthetic data
element_path = '/home/adrian/PhD/AxiSEM3D/CMB/kernels/forward/output__backup@2023-07-25T13:34:02/elements/test'

# location info
network = 'A'
station = '355'
location = '*'

# Window size [seconds]
T = 40
# Phase
phase = 'P'

###############################
# KERNEL COMPUTATION PARAMETERS
###############################

path_to_backward = '/home/adrian/PhD/AxiSEM3D/CMB/kernels/backward/output__backup@2023-07-24T20:03:43/elements/test'
path_to_inversion_mesh = '/disks/data/PhD/CMB/stations/STA_3D_UNIFORM_STA.txt'

################
# IMPLEMENTATION
################
real_data_obspyobj = ObspyfiedOutput(mseed_file_path=real_data_path)
inventory = real_data_obspyobj.inv
inventory = inventory.select(network=network, 
                                station=station, 
                                location=location)
station_depth = -inventory[0][0].elevation
station_latitude = inventory[0][0].latitude
station_longitude = inventory[0][0].longitude 
catalogue = real_data_obspyobj.cat
event_depth = catalogue[0].origins[0].depth * 1e-3  # must be in km for get_travel_times_geo
event_latitude = catalogue[0].origins[0].latitude
event_longitude = catalogue[0].origins[0].longitude

""" window_left, window_right = find_phase_window(event_depth, event_latitude, event_longitude, 
                                              station_latitude, station_longitude, T, phase) """
window_left = 600
window_right = 850

if SOURCE_BUILDING is True:
    L2_STF_builder(real_data_path, element_path, station, network, 
                  location, window_left, window_right)

if KERNEL_COMPUTATION is True:
    kernel = L2Kernel(element_path, path_to_backward)
    #kernel.evaluate_on_mesh(path_to_inversion_mesh, '/disks/data/PhD/AxiSEM3D-Kernels/KERNELS')
    kernel.evaluate_on_slice([0, 0, 0], [0, 0, 60], 
                             3000000, 6371000, 
                             theta_min=np.deg2rad(-10), theta_max=np.deg2rad(70),
                             N=200, slice_out_path='none', log_plot=False,
                             low_range=0, high_range=0.01, show_points=True)