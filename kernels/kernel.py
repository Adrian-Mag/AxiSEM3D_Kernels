from AxiSEM3D_Data_Handler.element_output import ElementOutput
from .helper_functions import window_data, sph2cart, cart2sph

import numpy as np
import pandas as pd
from scipy import integrate 
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import os 
from tqdm import tqdm


class L2Kernel():

    def __init__(self, forward_data_path, backward_data_path,
                 window_left, window_right):
        self.forward_data = ElementOutput(forward_data_path)
        self.backward_data = ElementOutput(backward_data_path)
        self.window_left = window_left
        self.window_right = window_right

        # get the forward and backward time 
        fw_time = self.forward_data.data_time
        self.fw_dt = fw_time[1] - fw_time[0]
        bw_time = self.backward_data.data_time
        self.bw_dt = bw_time[1] - bw_time[0]
        # Find the master time (minmax/maxmin)
        t_max = min(fw_time[-1], bw_time[-1])
        t_min = max(fw_time[0], bw_time[0])
        dt = max(self.fw_dt, self.bw_dt)
        self.master_time = np.arange(t_min, t_max, dt)

        self.fw_time = fw_time[0:-1] + self.fw_dt
        self.bw_time = bw_time[0:-1] + self.bw_dt


    def evaluate_on_mesh(self, path_to_inversion_mesh, sensitivity_out_path):
        # Earth's radius in m
        R = 6371000

        # import inversion mesh
        points = pd.read_csv(path_to_inversion_mesh, sep=" ")

        # initialize sensitivity 
        sensitivity = {'radius': [], 'latitude': [], 'longitude': [], 'sensitivity': []}

        for _, row in points.iterrows():
            latitude = row['latitude']
            longitude = row['longitude']
            radius = R - row['depth']

            # integrate over time the dot product/disks/data/PhD/CMB/simu1D_element/BACKWARD_UNIT_DELAY
            sensitivity['radius'].append(radius)
            sensitivity['latitude'].append(latitude)
            sensitivity['longitude'].append(longitude)
            sensitivity['sensitivity'].append(self.evaluate(radius, latitude, longitude))

    
        sensitivity_df = pd.DataFrame(sensitivity)
        sensitivity_df.to_csv(sensitivity_out_path + '/' + 'sensitivity_rho.txt', sep=' ', index=False)

    def evaluate_rho_0(self, point):

        # get forwards and backward waveforms at this point
        forward_waveform = np.nan_to_num(self.forward_data.load_data_at_point(point, channels=['U']))
        backward_waveform = np.nan_to_num(self.backward_data.load_data_at_point(point, channels=['U']))

        # Compute time derivatives wrt to time
        dfwdt = np.diff(forward_waveform) / self.fw_dt
        dbwdt = np.diff(backward_waveform) / self.bw_dt

        # Project both arrays on the master time
        interpolated_dfwdt = []
        interpolated_dbwdt = []

        for i in range(3):
            interpolated_dfwdt.append(np.interp(self.master_time, self.fw_time, dfwdt[i]))
            interpolated_dbwdt.append(np.interp(self.master_time, self.bw_time, dbwdt[i]))
        interpolated_dfwdt = np.array(interpolated_dfwdt)
        interpolated_dbwdt = np.array(interpolated_dbwdt)

        # flip backward waveform in the time axis
        reversed_dbwdt = np.flip(interpolated_dbwdt, axis=1)

        # Window the data
        if self.window_left is not None and self.window_right is not None:
            windowed_time, windowed_interp_dfwdt = window_data(self.master_time, interpolated_dfwdt, self.window_left, self.window_right)
            _, windowed_reversed_dbwdt = window_data(self.master_time, reversed_dbwdt, self.window_left, self.window_right)
        else:
            windowed_time = self.master_time
            windowed_interp_dfwdt = interpolated_dfwdt
            windowed_reversed_dbwdt = reversed_dbwdt

        # make dot product 
        fw_bw = (windowed_interp_dfwdt * windowed_reversed_dbwdt).sum(axis=0)
        
        sensitivity = integrate.simpson(fw_bw, dx=(windowed_time[1] - windowed_time[0]))
        return sensitivity
    

    def evaluate_lambda(self, point):
        # get forwards and backward waveforms at this point
        forward_waveform = np.nan_to_num(self.forward_data.load_data_at_point(point, channels=['E']))
        backward_waveform = np.nan_to_num(self.backward_data.load_data_at_point(point, channels=['E']))




    def evaluate_mu(self, point):
        # get forwards and backward waveforms at this point
        forward_waveform = np.nan_to_num(self.forward_data.load_data_at_point(point, channels=['E']))
        backward_waveform = np.nan_to_num(self.backward_data.load_data_at_point(point, channels=['E']))


    def evaluate_on_slice(self, source_loc: list, station_loc: list,
                          R_min: float, R_max: float, N: int, slice_out_path: str):
        
        filtered_indices, filtered_slice_points, \
        point1, point2, base1, base2, \
        inplane_DIM1, inplane_DIM2 = self.forward_data._create_slice(source_loc, station_loc, R_max=R_max,
                                                                        R_min=R_min, resolution=N, return_slice=True)
        # Initialize sensitivity values on the slice (Slice frame)
        inplane_sensitivity = np.zeros((N, N))
        
        with tqdm(total=len(filtered_slice_points)) as pbar:
            for [index1, index2], point in zip(filtered_indices, filtered_slice_points):
                inplane_sensitivity[index1, index2] = self.evaluate(point)
                pbar.update(1)
                    
        cbar_min, cbar_max = self._find_range(inplane_sensitivity, percentage_min=0.1, percentage_max=0.99)
        
        plt.figure()
        contour = plt.contourf(inplane_DIM1, inplane_DIM2, np.nan_to_num(inplane_sensitivity),
                     levels=np.linspace(cbar_min, cbar_max, 100), cmap='RdBu_r', extend='both')
        plt.scatter(np.dot(point1, base1), np.dot(point1, base2))
        plt.scatter(np.dot(point2, base1), np.dot(point2, base2))
        cbar = plt.colorbar(contour)

        cbar_ticks = np.linspace(int(cbar_min), int(cbar_max), 5) # Example tick values
        cbar_ticklabels = [str(cbar_tick) for cbar_tick in cbar_ticks] # Example tick labels
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels(cbar_ticklabels)
        cbar.set_label('Intensity')
        plt.show()


    def _find_range(self, arr, percentage_min, percentage_max):
        """
        Find the smallest value in the array based on the given percentage.

        Args:
            arr (ndarray): The input array.
            percentage (float): The percentage of values to consider.

        Returns:
            smallest_value (float or None): The smallest value based on the given percentage,
                                        or None if the array is empty or contains no finite values.
        """
        # Flatten the array to a 1D array
        flattened = arr[np.isfinite(arr)].flatten()
        
        if len(flattened) == 0:
            return None

        # Sort the flattened array in ascending order
        sorted_arr = np.sort(flattened)
        
        # Compute the index that corresponds to percentage of the values
        percentile_index_min = int(len(sorted_arr) * percentage_min)        
        percentile_index_max= int(len(sorted_arr) * percentage_max)
        
        # Get the value at the computed index
        smallest_value = sorted_arr[percentile_index_min]
        biggest_value = sorted_arr[percentile_index_max]
        
        return [smallest_value, biggest_value]   