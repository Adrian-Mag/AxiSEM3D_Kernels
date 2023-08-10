from ..handlers.element_output import ElementOutput
from ...aux.helper_functions import window_data
from ...aux.mesher import Mesh, SliceMesh

import numpy as np
import pandas as pd
from scipy import integrate 
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import os 
from tqdm import tqdm
import random


class Kernel():

    def __init__(self, forward_obj: ElementOutput, backward_obj: ElementOutput):
        self.forward_data = forward_obj
        self.backward_data = backward_obj

        # get the forward and backward time (assuming that all element groups
        # have the same time axis)
        first_group = next(iter(self.forward_data.element_groups_info))
        fw_time = self.forward_data.element_groups_info[first_group]['metadata']['data_time']
        self.fw_dt = fw_time[1] - fw_time[0]
        bw_time = self.backward_data.element_groups_info[first_group]['metadata']['data_time']
        # Apply again t -> T-t transform on the adjoint time
        bw_time = np.flip(np.max(bw_time) - bw_time)
        self.bw_dt = bw_time[1] - bw_time[0]

        # Check if the times
        # Find the master time (minmax/maxmin)
        t_max = min(fw_time[-1], bw_time[-1])
        t_min = max(fw_time[0], bw_time[0])
        dt = max(self.fw_dt, self.bw_dt)
        self.master_time = np.arange(t_min, t_max + dt, dt)

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


    def evaluate_rho_0(self, points: np.ndarray) -> np.ndarray:
        # get forwards and backward displacements at these points
        forward_waveform = np.nan_to_num(self.forward_data.load_data(points=points, channels=['U'], in_deg=False))
        backward_waveform = np.nan_to_num(self.backward_data.load_data(points=points, channels=['U'], in_deg=False))

        # Apply again t -> T-t transform on the adjoint data 
        backward_waveform = np.flip(backward_waveform, axis=2)
        # Compute time at the derivative points
        fw_time = self.fw_time[0:-1] + self.fw_dt / 2
        bw_time = self.bw_time[0:-1] + self.bw_dt / 2
        # Compute time derivatives wrt to time
        dfwdt = np.diff(forward_waveform, axis=2) / self.fw_dt
        dbwdt = np.diff(backward_waveform, axis=2) / self.bw_dt

        # Interpolate onto the master_time
        interp_dfwdt = np.empty(dfwdt.shape[:-1] + (len(self.master_time),))
        interp_dbwdt = np.empty(dbwdt.shape[:-1] + (len(self.master_time),))

        for i in range(dfwdt.shape[0]):
            for j in range(3):
                interp_dfwdt[i,j] = np.interp(self.master_time, fw_time, dfwdt[i,j])
                interp_dbwdt[i,j] = np.interp(self.master_time, bw_time, dbwdt[i,j])

        # make dot product 
        fw_bw = np.sum(interp_dfwdt * interp_dbwdt, axis=1)

        sensitivity = integrate.simpson(fw_bw, dx=(self.master_time[1] - self.master_time[0]))
        return sensitivity
    

    def evaluate_lambda(self, points: np.ndarray) -> np.ndarray:
        # K_lambda^zero = int_T (div u)(div u^t) = int_T (tr E)(tr E^t) = 
        # int_T (EZZ+ERR+ETT)(EZZ^t+ERR^t+ETT^t)

        # We first try to compute the sensitivity using the strain tensor, but
        # if it is not available, then we will use the gradient of displacement
        
        # get forwards and backward waveforms at this point
        forward_waveform = np.nan_to_num(self.forward_data.load_data(points, channels=['GZZ', 'GRR', 'GTT'], in_deg=False))
        backward_waveform = np.nan_to_num(self.backward_data.load_data(points, channels=['GZZ', 'GRR', 'GTT'], in_deg=False))

        #compute trace of each wavefield and flip adjoint in time
        trace_G = forward_waveform.sum(axis=1)
        trace_G_adjoint = np.flip(backward_waveform.sum(axis=1), axis=1)

        # Project both on master time
        interp_trace_G = np.empty(trace_G.shape[:-1] + (len(self.master_time),))
        interp_trace_G_adjoint = np.empty(trace_G.shape[:-1] + (len(self.master_time),))

        for i in range(len(points)):
            interp_trace_G[i] = np.interp(self.master_time, self.fw_time, trace_G[i])
            interp_trace_G_adjoint[i] = np.interp(self.master_time, self.bw_time, trace_G_adjoint[i])

        return integrate.simpson(interp_trace_G * interp_trace_G_adjoint, 
                                 dx = (self.master_time[1] - self.master_time[0]))


    def evaluate_mu(self, points: np.ndarray) -> np.ndarray:
        # K_mu_0 = int_T (grad u^t):(grad u) + (grad u^t):(grad u)^T 
        # = int_T 2E^t:E

        # We first try to compute the sensitivity using the strain tensor, but
        # if it is not available, then we will use the gradient of displacement

        # get forwards and backward waveforms at this point
        G_forward = np.nan_to_num(self.forward_data.load_data(points, channels=['G']))
        G_adjoint = np.nan_to_num(self.backward_data.load_data(points, channels=['G']))

        # flip adjoint in time
        G_adjoint = np.flip(G_adjoint, axis=2)
        
        # Project both arrays on the master time
        interp_G_forward = np.empty(G_forward.shape[:-1] + (len(self.master_time),))
        interp_G_adjoint = np.empty(G_adjoint.shape[:-1] + (len(self.master_time),))
        for i in range(len(points)):
            for j in range(9):
                interp_G_forward[i,j] = np.interp(self.master_time, self.fw_time, G_forward[i,j])
                interp_G_adjoint[i,j] = np.interp(self.master_time, self.bw_time, G_adjoint[i,j])
        interp_G_forward = interp_G_forward.reshape(len(points), 3, 3, len(self.master_time))
        interp_G_adjoint = interp_G_adjoint.reshape(len(points), 3, 3, len(self.master_time))

        # Multiply
        integrand = np.sum(interp_G_adjoint * (interp_G_forward + interp_G_forward.transpose(0,2,1,3)), axis=(1,2))

        return integrate.simpson(integrand, dx = (self.master_time[1] - self.master_time[0]))


    def evaluate_rho(self, points: np.ndarray) -> np.ndarray:
        # K_rho = K_rho_0 + (vp^2-2vs^2)K_lambda_0 + vs^2 K_mu_0
        if self.forward_data.base_model['type'] == 'axisem3d':
            radii = np.array(self.forward_data.base_model['R'])
        else:
            radii = np.array(self.forward_data.base_model['DATA']['radius'])
        is_increasing = radii[0] < radii[1]
        if is_increasing:
            indices = np.searchsorted(radii, points[:,0])
        else:
            indices = np.searchsorted(-radii, -points[:,0])

        if self.forward_data.base_model['type'] == 'axisem3d':
            vp = self.forward_data.base_model['VP'][indices - 1]
            vs = self.forward_data.base_model['VS'][indices - 1]
        else:
            vp = np.array(self.forward_data.base_model['DATA']['vp'])[indices - 1]
            vs = np.array(self.forward_data.base_model['DATA']['vs'])[indices - 1]

        return self.evaluate_rho_0(points) + (vp*vp - 2*vs*vs)*self.evaluate_lambda(points) + vs*vs*self.evaluate_mu(points)


    def evaluate_vs(self, point):
        # K_vs = 2*rho*vs*(K_mu_0 - 2*K_lambda_0)
        radii = np.array(self.forward_data.base_model['R'])
        is_increasing = radii[0] < radii[1]
        if is_increasing:
            index = np.searchsorted(radii, point[0])
        else:
            index = np.searchsorted(-radii, -point[0])
        if index == 0 or index >= len(radii):
            print('Point outside of base model domain')
            return 1
        else:
            rho = self.forward_data.base_model['RHO'][index - 1]
            vs = self.forward_data.base_model['VS'][index - 1]
        return 2 * rho * vs * (self.evaluate_mu_0(point) - 2*self.evaluate_mu_0(point))
    
    
    def evaluate_vp(self, point):
        # K_vs = 2*rho*vp*K_lambda_0
        radii = np.array(self.forward_data.base_model['R'])
        is_increasing = radii[0] < radii[1]
        if is_increasing:
            index = np.searchsorted(radii, point[0])
        else:
            index = np.searchsorted(-radii, -point[0])
        if index == 0 or index >= len(radii):
            print('Point outside of base model domain')
            return 1
        else:
            rho = self.forward_data.base_model['RHO'][index - 1]
            vp = self.forward_data.base_model['VP'][index - 1]
        return 2 * rho * vp * self.evaluate_lambda_0(point)


    def evaluate_on_slice(self, source_location: list=None, station_location: list=None,
                          resolution: int=50, domains: list=None,
                          log_plot: bool=False, low_range: float=0.1, high_range: float=0.999):
        # Create default domains if None were given
        if domains is None:
            domains = []
            for element_group in self.forward_data.element_groups_info.values():
                domains.append(element_group['elements']['vertical_range'] + [-2*np.pi, 2*np.pi])
        domains = np.array(domains)

        # Create source and station if none were given
        R_max = np.max(domains[:,1])
        R_min = np.min(domains[:,0])
        if source_location is None and station_location is None:
            source_location = np.array([R_max, 
                                        np.radians(self.forward_data.source_lat), 
                                        np.radians(self.forward_data.source_lon)])
            station_location = np.array([R_max, 
                                         np.radians(self.backward_data.source_lat), 
                                         np.radians(self.backward_data.source_lon)])

        # Create e slice mesh
        mesh = SliceMesh(source_location, station_location, domains, resolution)
        
        # Compute sensitivity values on the slice (Slice frame)
        inplane_sensitivity = np.full((mesh.resolution, mesh.resolution), fill_value=np.NaN)
        data = self.evaluate_rho_0(mesh.points)
        # Distribute the values in the matrix that will be plotted
        index = 0
        for [index1, index2], _ in zip(mesh.indices, mesh.points):
            inplane_sensitivity[index1, index2] = data[index]
            index += 1

        if log_plot is False:
            _, cbar_max = self._find_range(inplane_sensitivity, percentage_min=0, percentage_max=1)
            cbar_max *= (high_range * high_range)
            cbar_min = -cbar_max
            plt.figure()
            contour = plt.contourf(mesh.inplane_DIM1, mesh.inplane_DIM2, np.nan_to_num(inplane_sensitivity),
                        levels=np.linspace(cbar_min, cbar_max, 100), cmap='RdBu_r', extend='both')
        else:
            cbar_min, cbar_max = self._find_range(np.log10(np.abs(inplane_sensitivity)), percentage_min=low_range, percentage_max=high_range)
            
            plt.figure()
            contour = plt.contourf(mesh.inplane_DIM1, mesh.inplane_DIM2, np.log10(np.abs(inplane_sensitivity)),
                        levels=np.linspace(cbar_min, cbar_max, 100), cmap='RdBu_r', extend='both')
        plt.scatter(np.dot(mesh.point1, mesh.base1), np.dot(mesh.point1, mesh.base2))
        plt.scatter(np.dot(mesh.point2, mesh.base1), np.dot(mesh.point2, mesh.base2))
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