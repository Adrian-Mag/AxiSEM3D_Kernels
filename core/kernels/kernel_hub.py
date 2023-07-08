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
real_data_path = '/disks/data/PhD/CMB/simu3D_CMB/REAL_DATA/output/stations/Station_grid/obspyfied/Station_grid.mseed'
# Path to element output of synthetic data
element_path = '/disks/data/PhD/CMB/simu1D_element/FORWARD/output/elements/forward_20s_1D'

# location info
network = 'A'
station = '356'
location = '*'

# Window size [seconds]
T = 40
# Phase
phase = 'P'

###############################
# KERNEL COMPUTATION PARAMETERS
###############################

path_to_backward = '/disks/data/PhD/CMB/simu1D_element_backward/BACKWARD/output/elements/entire_earth'
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

window_left, window_right = find_phase_window(event_depth, event_latitude, event_longitude, 
                                              station_latitude, station_longitude, T, phase)
#window_left = None
#window_right = None

if SOURCE_BUILDING is True:
    L2_STF_builder(real_data_path, element_path, station, network, 
                  location, window_left, window_right)

if KERNEL_COMPUTATION is True:
    kernel = L2Kernel(element_path, path_to_backward)
    #kernel.evaluate_on_mesh(path_to_inversion_mesh, '/disks/data/PhD/AxiSEM3D-Kernels/KERNELS')
    kernel.evaluate_on_slice([0, 0, 0], [0, 0, 10], 
                             3480000, 6371000, 
                             theta_min=np.deg2rad(-10), theta_max=np.deg2rad(50),
                             N=300, slice_out_path='none', log_plot=False,
                             low_range=0, high_range=0.1, show_points=False)