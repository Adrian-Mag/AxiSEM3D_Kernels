from ..handlers.element_output import ElementOutput
from ...aux.helper_functions import window_data

import numpy as np
import pandas as pd
from scipy import integrate 
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import os 
from tqdm import tqdm
import random


class L2Kernel():

    def __init__(self, forward_data_path, backward_data_path):
        self.forward_data = ElementOutput(forward_data_path)
        self.backward_data = ElementOutput(backward_data_path)

        # get the forward and backward time 
        fw_time = self.forward_data.data_time
        self.fw_dt = fw_time[1] - fw_time[0]
        bw_time = self.backward_data.data_time
        # Apply again t -> T-t transform on the adjoint time
        bw_time = np.flip(np.max(bw_time) - bw_time)
        self.bw_dt = bw_time[1] - bw_time[0]

        # Find the master time (minmax/maxmin)
        t_max = min(fw_time[-1], bw_time[-1])
        t_min = max(fw_time[0], bw_time[0])
        dt = max(self.fw_dt, self.bw_dt)
        self.master_time = np.arange(t_min, t_max, dt)

        self.fw_time = fw_time
        self.bw_time = bw_time


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
        
        # Apply again t -> T-t transform on the adjoint data 
        backward_waveform = np.flip(backward_waveform)
        # Compute time at the derivative points
        fw_time = self.fw_time[0:-1] + self.fw_dt / 2
        bw_time = self.bw_time[0:-1] + self.bw_dt / 2
        # Compute time derivatives wrt to time
        dfwdt = np.diff(forward_waveform) / self.fw_dt
        dbwdt = np.diff(backward_waveform) / self.bw_dt

        # Project both arrays on the master time
        interp_dfwdt = []
        interp_dbwdt = []

        for i in range(3):
            interp_dfwdt.append(np.interp(self.master_time, fw_time, dfwdt[i]))
            interp_dbwdt.append(np.interp(self.master_time, bw_time, dbwdt[i]))
        interp_dfwdt = np.array(interp_dfwdt)
        interp_dbwdt = np.array(interp_dbwdt)

        # make dot product 
        fw_bw = (interp_dfwdt * interp_dbwdt).sum(axis=0)

        sensitivity = integrate.simpson(fw_bw, dx=(self.master_time[1] - self.master_time[0]))
        return sensitivity
    

    def evaluate_lambda_0(self, point):
        # K_lambda^zero = int_T (div u)(div u^t) = int_T (tr E)(tr E^t) = 
        # int_T (EZZ+ERR+ETT)(EZZ^t+ERR^t+ETT^t)

        # get forwards and backward waveforms at this point
        forward_waveform = np.nan_to_num(self.forward_data.load_data_at_point(point, channels=['EZZ', 'ERR', 'ETT']))
        backward_waveform = np.nan_to_num(self.backward_data.load_data_at_point(point, channels=['EZZ', 'ERR', 'ETT']))

        #compute trace of each wavefield and flip adjoint in time
        trace_E = forward_waveform.sum(axis=0)
        trace_E_adjoint = np.flip(backward_waveform.sum(axis=0))

        # Project both on master time
        interp_trace_E = []
        interp_trace_E_adjoint = []

        interp_trace_E.append(np.interp(self.master_time, self.fw_time, trace_E))
        interp_trace_E_adjoint.append(np.interp(self.master_time, self.bw_time, trace_E_adjoint))
        
        interp_trace_E = np.array(interp_trace_E)
        interp_trace_E_adjoint = np.array(interp_trace_E_adjoint)

        return integrate.simpson(interp_trace_E * interp_trace_E_adjoint, 
                                 dx = (self.master_time[1] - self.master_time[0]))


    def evaluate_mu_0(self, point):
        # K_mu_0 = int_T (grad u^t):(grad u) + (grad u^t):(grad u)^T 
        # = int_T 2E^t:E

        # get forwards and backward waveforms at this point
        E = np.nan_to_num(self.forward_data.load_data_at_point(point, channels=['E']))
        E_adjoint = np.nan_to_num(self.backward_data.load_data_at_point(point, channels=['E']))

        # flip adjoint in time
        E_adjoint = np.flip(E_adjoint)
        
        # Project both arrays on the master time
        interp_E = []
        interp_E_adjoint = []
        for i in range(6):
            interp_E.append(np.interp(self.master_time, self.fw_time, E[i]))
            interp_E_adjoint.append(np.interp(self.master_time, self.bw_time, E_adjoint[i]))
        interp_E = np.array(interp_E)
        interp_E_adjoint = np.array(interp_E_adjoint)
            
        weights = np.array([1, 1, 1, 2, 2, 2])
        # Multiply 
        integrand = 2 * np.sum((interp_E_adjoint * interp_E) * weights[:, np.newaxis], axis=1)

        return integrate.simpson(integrand, dx = (self.master_time[1] - self.master_time[0]))

    def evaluate_on_slice(self, source_loc: list, station_loc: list,
                          R_min: float, R_max: float, theta_min: float, theta_max: float, 
                          N: int, slice_out_path: str, show_points: bool=False,
                          log_plot: bool=False, low_range: float=0.1, high_range: float=0.999):
        
        filtered_indices, filtered_slice_points, \
        point1, point2, base1, base2, \
        inplane_DIM1, inplane_DIM2 = self.forward_data._create_slice(source_loc, station_loc, R_max=R_max, theta_min=theta_min, theta_max=theta_max,
                                                                        R_min=R_min, resolution=N, return_slice=True)
        # Initialize sensitivity values on the slice (Slice frame)
        inplane_sensitivity = np.zeros((N, N))
        
        with tqdm(total=len(filtered_slice_points)) as pbar:
            for [index1, index2], point in zip(filtered_indices, filtered_slice_points):
                inplane_sensitivity[index1, index2] = self.evaluate_rho_0(point)
                pbar.update(1)
        
        if log_plot is False:
            _, cbar_max = self._find_range(inplane_sensitivity, percentage_min=0, percentage_max=1)
            cbar_max *= (high_range * high_range)
            cbar_min = -cbar_max
            plt.figure()
            contour = plt.contourf(inplane_DIM1, inplane_DIM2, np.nan_to_num(inplane_sensitivity),
                        levels=np.linspace(cbar_min, cbar_max, 100), cmap='RdBu_r', extend='both')
        else:
            cbar_min, cbar_max = self._find_range(np.log10(np.abs(inplane_sensitivity)), percentage_min=low_range, percentage_max=high_range)
            
            plt.figure()
            contour = plt.contourf(inplane_DIM1, inplane_DIM2, np.log10(np.abs(inplane_sensitivity)),
                        levels=np.linspace(cbar_min, cbar_max, 100), cmap='RdBu_r', extend='both')
        if show_points:
            plt.scatter(np.dot(point1, base1), np.dot(point1, base2))
            plt.scatter(np.dot(point2, base1), np.dot(point2, base2))
        cbar = plt.colorbar(contour)

        cbar_ticks = np.linspace(cbar_min, cbar_max, 5) # Example tick values
        cbar_ticklabels = ["{:.2e}".format(cbar_tick) for cbar_tick in cbar_ticks] # Example tick labels
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
        percentile_index_min = int((len(sorted_arr)-1) * percentage_min)        
        percentile_index_max= int((len(sorted_arr)-1) * percentage_max)
        
        # Get the value at the computed index
        smallest_value = sorted_arr[percentile_index_min]
        biggest_value = sorted_arr[percentile_index_max]
        
        return [smallest_value, biggest_value]   

    
    def _point_in_region_of_interest(self, point)-> bool:
        if random.random() < 0.01:
            max_lat = 30
            min_lat = -30
            min_lon = -30
            max_lon = 30
            min_rad = 3400000
            max_rad = 6371000
            if point[0] < max_rad and point[0] > min_rad \
                and np.rad2deg(point[1]) < max_lat and np.rad2deg(point[1]) > min_lat \
                and np.rad2deg(point[2]) < max_lon and np.rad2deg(point[2]) > min_lon:
                return True
            else:
                return False
        else:
            return False