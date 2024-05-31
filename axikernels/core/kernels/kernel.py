from ..handlers.element_output import ElementOutput
from ...aux.helper_functions import window_data
from ...aux.mesher import Mesh, SliceMesh, SphereMesh
from ...aux.coordinate_transforms import sph2cart_mpmath, sph2cart, cart2sph, cart2sph_mpmath, cart2polar, cart_geo2cart_src, cart2cyl


import numpy as np
import pandas as pd
from scipy import integrate
import matplotlib
matplotlib.use('Qtagg')
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import random
from mayavi import mlab
from scipy.interpolate import interp2d
from scipy.interpolate import RectBivariateSpline
from mpl_toolkits.basemap import Basemap
from matplotlib import cm


class Kernel():

    def __init__(self, forward_obj: ElementOutput, backward_obj: ElementOutput):
        # We store the forward and backward data
        self.forward_data = forward_obj
        self.backward_data = backward_obj

        # get the forward and backward time (assuming that all element groups
        # have the same time axis!!)
        first_group = self.forward_data.element_groups[0]
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
        forward_waveform = np.nan_to_num(self.forward_data.load_data(points=points, channels=['UR','UT','UZ'], in_deg=False))
        backward_waveform = np.nan_to_num(self.backward_data.load_data(points=points, channels=['UR','UT','UZ'], in_deg=False))

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
        G_forward = np.nan_to_num(self.forward_data.load_data(points, channels=['GRR','GRT','GRZ','GTR','GTT','GTZ','GZR','GZT','GZZ'], in_deg=False))
        G_adjoint = np.nan_to_num(self.backward_data.load_data(points, channels=['GRR','GRT','GRZ','GTR','GTT','GTZ','GZR','GZT','GZZ'], in_deg=False))

        # flip adjoint in time
        G_adjoint = np.flip(G_adjoint, axis=2)

        # Project both arrays on the master time
        interp_G_forward = np.empty(G_forward.shape[:-1] + (len(self.master_time),))
        interp_G_adjoint = np.empty(G_adjoint.shape[:-1] + (len(self.master_time),))
        for i in range(len(points)):
            for j in range(9):
                interp_G_forward[i,j,:] = np.interp(self.master_time, self.fw_time, G_forward[i,j,:])
                interp_G_adjoint[i,j,:] = np.interp(self.master_time, self.bw_time, G_adjoint[i,j,:])

        interp_G_forward = interp_G_forward.reshape(len(points), 3, 3, len(self.master_time))
        interp_G_adjoint = interp_G_adjoint.reshape(len(points), 3, 3, len(self.master_time))

        # Multiply
        integrand = np.sum(interp_G_adjoint * (interp_G_forward + interp_G_forward.transpose(0,2,1,3)), axis=(1,2))

        return integrate.simpson(integrand, dx = (self.master_time[1] - self.master_time[0]))


    def evaluate_rho(self, points: np.ndarray) -> np.ndarray:
        # K_rho = K_rho_0 + (vp^2-2vs^2)K_lambda_0 + vs^2 K_mu_0
        radii = np.array(self.forward_data.base_model['DATA']['radius'])
        is_increasing = radii[0] < radii[1]
        if is_increasing:
            index = np.searchsorted(radii, points[:,0])
        else:
            index = np.searchsorted(-radii, -points[:,0])
        # eliminated points outside of the domain
        mask = np.logical_or(index > 0, index < len(radii))
        filtered_index = index[mask]

        rho = np.array(self.forward_data.base_model['DATA']['rho'])[filtered_index - 1]
        vp = np.array(self.forward_data.base_model['DATA']['vp'])[filtered_index - 1]
        vs = np.array(self.forward_data.base_model['DATA']['vs'])[filtered_index - 1]

        return self.evaluate_rho_0(points) + (vp*vp - 2*vs*vs)*self.evaluate_lambda(points) + vs*vs*self.evaluate_mu(points)


    def evaluate_vp(self, points):
        # K_vs = 2*rho*vp*K_lambda_0
        radii = np.array(self.forward_data.base_model['DATA']['radius'])
        is_increasing = radii[0] < radii[1]
        if is_increasing:
            index = np.searchsorted(radii, points[:,0])
        else:
            index = np.searchsorted(-radii, -points[:,0])
        # eliminated points outside of the domain
        mask = np.logical_or(index > 0, index < len(radii))
        filtered_index = index[mask]

        rho = np.array(self.forward_data.base_model['DATA']['rho'])[filtered_index - 1]
        vp = np.array(self.forward_data.base_model['DATA']['vp'])[filtered_index - 1]

        return 2 * rho * vp * self.evaluate_lambda(points)


    def evaluate_vs(self, points):
        # K_vs = 2*rho*vs*(K_mu_0 - 2*K_lambda_0)
        radii = np.array(self.forward_data.base_model['DATA']['radius'])
        is_increasing = radii[0] < radii[1]
        if is_increasing:
            index = np.searchsorted(radii, points[:,0])
        else:
            index = np.searchsorted(-radii, -points[:,0])
        # eliminated points outside of the domain
        mask = np.logical_or(index > 0, index < len(radii))
        filtered_index = index[mask]

        rho = np.array(self.forward_data.base_model['DATA']['rho'])[filtered_index - 1]
        vs = np.array(self.forward_data.base_model['DATA']['vs'])[filtered_index - 1]

        return 2 * rho * vs * (self.evaluate_mu(points) - 2*self.evaluate_lambda(points))


    def evaluate_geometric(self, points: np.ndarray, radius: float):
        # All points must be in the format  (lat, lon) in radians, where lat and
        # lon are in the geographical frame. The radius must be in meters. This
        # only works for solid-solid now
        pass


    def _find_discontinuity_type(self, radius: float):
        # Find the desired discontinuity in the base model (for 1D only)
        if radius in self.forward_data.base_model['DISCONTINUITIES']:
            # Find what type of discontinuity it is (only SS works for now)
            radius_index = self.forward_data.base_model['DATA']['radius'].index(radius)
            [vs_upper, vs_lower] = np.array(self.forward_data.base_model['DATA']['vs'])[[radius_index, radius_index+1]]
            if vs_upper > 0 and vs_lower > 0:
                return 'SS'
            elif vs_upper > 0 and vs_lower == 0:
                return 'FS'
            elif vs_upper == 0 and vs_lower > 0:
                return 'SF'
        else:
            raise ValueError('There is no discontinuity at {}. \
                             The available discontinuities are at {}'.format(radius,
                                                                             self.forward_data.base_model['DISCONTINUITIES'])
                            )


    def _form_limit_points(self, points: np.ndarray, radius: float) -> np.ndarray:
        # Form upper and lower limit points
        dr = 1000
        upper_points = np.array([[radius + dr, lat, lon] for lat, lon in points])
        lower_points = np.array([[radius - dr, lat, lon] for lat, lon in points])

        return (upper_points, lower_points)


    def evaluate_on_sphere(self, resolution: int, radius: int):
        # Compute points on spherical mesh
        # Define the lat lon grid (must match data file)
        lat = np.arange(-45, 45.01, 1)*np.pi/180
        lon = np.arange(-40, 120.01, 1)*np.pi/180
        LON, LAT = np.meshgrid(lon, lat)
        nlat = len(lat)
        nlon = len(lon)
        points = np.dstack((LAT,LON)).reshape(-1,2)

        # Compute kernel
        batch_size = 10000
        num_batches = len(points) // batch_size
        kernel = np.array([])
        """ for index in range(num_batches):
            kernel = np.append(kernel, self.evaluate_CMB_solid(points[index*batch_size:(index+1)*batch_size], radius=radius) -
                               self.evaluate_CMB_fluid(points[index*batch_size:(index+1)*batch_size], radius=radius))
        kernel = np.append(kernel, self.evaluate_CMB_solid(points[num_batches*batch_size:], radius=radius)-
                           self.evaluate_CMB_fluid(points[num_batches*batch_size:], radius=radius)) """

        """ for index in range(num_batches):
            kernel = np.append(kernel, self.evaluate_K_dv(points[index*batch_size:(index+1)*batch_size], radius=radius) +
                               self.evaluate_K_dn(points[index*batch_size:(index+1)*batch_size], radius=radius))
        kernel = np.append(kernel, self.evaluate_K_dv(points[num_batches*batch_size:], radius=radius) +
                           self.evaluate_K_dn(points[num_batches*batch_size:], radius=radius))
        np.savetxt('original_kernel.csv', kernel, delimiter=',') """

        kernel = np.loadtxt('/home/adrian/PhD/AxiSEM3D/moho.csv', delimiter=' ')

        kernel = kernel.reshape(LON.shape)

        # Define the lat lon grid for the larger mesh covering the Earth's surface
        larger_lat = np.arange(-90, 90.01, 1) * np.pi / 180
        larger_lon = np.arange(-180, 180.01, 1) * np.pi / 180
        # Create a larger mesh using the larger_lat and larger_lon
        LARGER_LON, LARGER_LAT = np.meshgrid(larger_lon, larger_lat)
        # Create an interpolation function based on the computed kernel
        interp_kernel = RectBivariateSpline(lat, lon, kernel)
        # Interpolate the kernel onto the larger mesh
        interpolated_kernel = interp_kernel(larger_lat, larger_lon)
        # Set the regions with missing data to a default value of 0
        interpolated_kernel[np.isnan(interpolated_kernel)] = 0
        # Construct CMB and Surface matrices
        R_disc = radius*1e-3*np.ones(np.shape(LARGER_LON))

        X_disc = R_disc * np.cos(LARGER_LAT) * np.cos(LARGER_LON)
        Y_disc = R_disc * np.cos(LARGER_LAT) * np.sin(LARGER_LON)
        Z_disc = R_disc * np.sin(LARGER_LAT)


        # create colormap
        N = len(interpolated_kernel.flatten()) # Number of points
        scalars = np.arange(N).reshape(interpolated_kernel.shape[0], interpolated_kernel.shape[1]) # Key point: set an integer for each point

        # Choose a colormap from matplotlib (e.g., 'viridis', 'plasma', 'cividis', etc.)
        cmap = cm.get_cmap('RdBu_r')  # Change this to your desired colormap

        # Calculate the color values based on the interpolated_kernel data
        gamma = 1.1
        max_range = interpolated_kernel.max() ** gamma
        color_values = (interpolated_kernel + max_range) / (2*max_range)

        # Convert color values to RGBA using the chosen colormap
        colors = (cmap(color_values) * 255).astype(np.uint8)

        # Set alpha channel to 255 (no transparency)
        colors[:, :, 3] = 255
        # Reshape the colors array to be 2D
        colors_reshaped = colors.reshape(-1, 4)  # Reshape to (N, 4) where N is the total number of points

        mlab.figure(bgcolor=(0,0,0))
        # Plot interpolated_kernel
        interpolated_kernel_surface = mlab.mesh(X_disc, Y_disc, Z_disc, scalars=scalars, mode='sphere', opacity=1)
        # Set look-up table and redraw
        interpolated_kernel_surface.module_manager.scalar_lut_manager.lut.table = colors_reshaped

        # Create Basemap instance
        m = Basemap(projection='cyl', llcrnrlat=-90, urcrnrlat=90,
                    llcrnrlon=-180, urcrnrlon=180, resolution='c')

        """  # Plot continent contours
        m.drawcoastlines(linewidth=0.5)

        # Get continent boundary polygons
        polys = m.landpolygons

        # Plot each continent polygon
        for polygon in polys:
            coords = np.array(polygon.get_coords())
            lon = coords[:, 0]
            lat = coords[:, 1]
            r = 6371 * np.ones(len(lon))
            x = r * np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(lon))
            y = r * np.cos(np.deg2rad(lat)) * np.sin(np.deg2rad(lon))
            z = r * np.sin(np.deg2rad(lat))
            mlab.plot3d(x, y, z, color=(0.8, 0.8, 0.8), tube_radius = 10) """

        mlab.show()


    def evaluate_CMB_solid(self, points: np.ndarray, radius: float) -> np.ndarray:
        # Get limit points
        upper_points, lower_points = self._form_limit_points(points, radius)

        # Find rho_upper and lower (assuming radius is in decreasing order)
        radius_index = self.forward_data.base_model['DATA']['radius'].index(radius)
        rho_upper, rho_lower = np.array(self.forward_data.base_model['DATA']['rho'])[[radius_index, radius_index + 1]]

        # COMPUTE IN SOLID

        # Load U, U', T, T'zz T'tz T'zr, Grr Gtt Gzz Gzr Gzt, G'rr G'tt G'rt G'tr G'rz G'tz
        U = self.forward_data.load_data(upper_points, frame='geographic', coords='spherical', in_deg=False,
                                        channels=['UR','UT','UZ'])
        T = self.forward_data.load_data(upper_points, frame='geographic', coords='spherical', in_deg=False,
                                        channels=['SRR','STT','SZZ','STZ','SZR','SRT'])
        G = self.forward_data.load_data(upper_points, frame='geographic', coords='spherical', in_deg=False,
                                        channels=['GRR','GTT','GZZ','GZR','GZT'])
        U_back = self.backward_data.load_data(upper_points, frame='geographic', coords='spherical', in_deg=False,
                                        channels=['UR','UT','UZ'])
        T_back = self.backward_data.load_data(upper_points, frame='geographic', coords='spherical', in_deg=False,
                                        channels=['SZZ','STZ','SZR'])
        G_back = self.backward_data.load_data(upper_points, frame='geographic', coords='spherical', in_deg=False,
                                        channels=['GRR','GTT','GRT','GTR','GRZ','GTZ'])

        # Flip adjoint values
        U_back = np.flip(U_back, axis=2)
        T_back = np.flip(T_back, axis=2)
        G_back = np.flip(G_back, axis=2)

        # Interpolate onto the master_time
        interp_U = np.empty(U.shape[:-1] + (len(self.master_time),))
        interp_T = np.empty(T.shape[:-1] + (len(self.master_time),))
        interp_G = np.empty(G.shape[:-1] + (len(self.master_time),))
        interp_U_back = np.empty(U_back.shape[:-1] + (len(self.master_time),))
        interp_T_back = np.empty(T_back.shape[:-1] + (len(self.master_time),))
        interp_G_back = np.empty(G_back.shape[:-1] + (len(self.master_time),))

        for i in range(len(points)):
            for j in range(3):
                interp_U[i,j] = np.interp(self.master_time, self.fw_time, U[i,j])
                interp_U_back[i,j] = np.interp(self.master_time, self.bw_time, U_back[i,j])
                interp_T_back[i,j] = np.interp(self.master_time, self.bw_time, T_back[i,j])
            for j in range(5):
                interp_G[i,j] = np.interp(self.master_time, self.fw_time, G[i,j])
            for j in range(6):
                interp_T[i,j] = np.interp(self.master_time, self.fw_time, T[i,j])
                interp_G_back[i,j] = np.interp(self.master_time, self.bw_time, G_back[i,j])

        # Compute K_rho_0 in solid
        # Compute time at the derivative points
        master_time_dt = self.master_time[1] - self.master_time[0]
        # Compute time derivatives wrt to time
        dU_dt = np.diff(interp_U, axis=2) / master_time_dt
        dU_back_dt = np.diff(interp_U_back, axis=2) / master_time_dt
        # make dot product
        fw_bw = np.sum(dU_dt * dU_back_dt, axis=1)
        K_rho_0 = rho_upper * integrate.simpson(fw_bw, dx=master_time_dt)

         # Compute rest of Kdv + K_dn in solid
        integrand = interp_T[:,0,:] * interp_G_back[:,0,:] + interp_T[:,1,:] * interp_G_back[:,1,:] + \
                interp_T[:,5,:] * (interp_G_back[:,2,:] + interp_G[:,3,:]) + \
                interp_T[:,4,:] * interp_G_back[:,4,:] + interp_T[:,3,:] * interp_G_back[:,5,:] - \
                interp_T_back[:,2,:] * interp_G[:,3,:] - interp_T_back[:,1,:] * interp_G[:,4,:] - interp_T_back[:,0,:] * interp_G[:,2,:]

        K_dv_dn = integrate.simpson(integrand, dx=master_time_dt)

        """ # Compute tangential derivative meshes
        dtheta = 0.05*np.pi/180

        upper_points_source = cart2sph_mpmath(cart_geo2cart_src(sph2cart_mpmath(upper_points),
                                                         rotation_matrix=self.forward_data.rotation_matrix))
        upper_points_dR_source = upper_points_source.copy()
        upper_points_dR_source[:,1] -= dtheta
        upper_points_dT_source = upper_points_source.copy()
        upper_points_dT_source[:,2] -= dtheta
        upper_points_dR = cart2sph_mpmath(cart_geo2cart_src(sph2cart_mpmath(upper_points_dR_source),
                                                     rotation_matrix=self.forward_data.rotation_matrix.transpose()))
        upper_points_dT = cart2sph_mpmath(cart_geo2cart_src(sph2cart_mpmath(upper_points_dT_source),
                                                     rotation_matrix=self.forward_data.rotation_matrix.transpose()))

        # Load TZZ at new meshes
        TZZ_dR = self.forward_data.load_data(upper_points_dR,frame='geographic',
                                             coords='spherical', in_deg=False, channels=['SZZ'])
        TZZ_dT = self.forward_data.load_data(upper_points_dT,frame='geographic',
                                             coords='spherical', in_deg=False, channels=['SZZ'])
        TZZ_dR_back = self.backward_data.load_data(upper_points_dR,frame='geographic',
                                                   coords='spherical', in_deg=False, channels=['SZZ'])
        TZZ_dT_back = self.backward_data.load_data(upper_points_dT,frame='geographic',
                                                   coords='spherical', in_deg=False, channels=['SZZ'])
        interp_TZZ_dR = np.empty(TZZ_dR.shape[:-1] + (len(self.master_time),))
        interp_TZZ_dT = np.empty(TZZ_dT.shape[:-1] + (len(self.master_time),))
        interp_TZZ_dR_back = np.empty(TZZ_dR_back.shape[:-1] + (len(self.master_time),))
        interp_TZZ_dT_back = np.empty(TZZ_dR_back.shape[:-1] + (len(self.master_time),))
        for i in range(len(points)):
            interp_TZZ_dR[i,0] = np.interp(self.master_time, self.fw_time, TZZ_dR[i,0])
            interp_TZZ_dT[i,0] = np.interp(self.master_time, self.fw_time, TZZ_dT[i,0])
            interp_TZZ_dR_back[i,0] = np.interp(self.master_time, self.bw_time, TZZ_dR_back[i,0])
            interp_TZZ_dT_back[i,0] = np.interp(self.master_time, self.bw_time, TZZ_dT_back[i,0])

        dTZZ_dR = (interp_TZZ_dR[:,0,:] - interp_T[:,2,:]) / (dtheta * radius)
        dTZZ_dT = (interp_TZZ_dT[:,0,:] - interp_T[:,2,:]) / (dtheta * radius) * np.sin(upper_points[:,1])[:, np.newaxis]
        dTZZ_dR_back = (interp_TZZ_dR_back[:,0,:] - interp_T_back[:,2,:]) / (dtheta * radius)
        dTZZ_dT_back = (interp_TZZ_dT_back[:,0,:] - interp_T_back[:,2,:]) / (dtheta * radius) * np.sin(upper_points[:,1])[:, np.newaxis]

        integrand = interp_T[:,2,:] * (interp_G_back[:,0,:] + interp_G_back[:,1,:]) - \
                interp_T_back[:,0,:] * (interp_G[:,0,:] + interp_G[:,1,:]) -\
                dTZZ_dR * interp_U_back[:,0,:] - dTZZ_dT * interp_U_back[:,1,:] - \
                dTZZ_dR_back * interp_U[:,0,:] - dTZZ_dT_back * interp_U[:,1,:]

        K_d_sigma = integrate.simpson(integrand, dx=master_time_dt) """

        return K_rho_0 + K_dv_dn #+ K_d_sigma


    def evaluate_CMB_fluid(self, points: np.ndarray, radius: float) -> np.ndarray:
        # Get limit points
        upper_points, lower_points = self._form_limit_points(points, radius)

        # Find rho_upper and lower (assuming radius is in decreasing order)
        radius_index = self.forward_data.base_model['DATA']['radius'].index(radius)
        rho_upper, rho_lower = np.array(self.forward_data.base_model['DATA']['rho'])[[radius_index, radius_index + 1]]

        # COMPUTE IN FLUID

        # Load U, U', P, P', Grr Gtt Gzz
        U = self.forward_data.load_data(lower_points, frame='geographic', coords='spherical', in_deg=False,
                                        channels=['UR','UT','UZ'])
        P = self.forward_data.load_data(lower_points, frame='geographic', coords='spherical', in_deg=False,
                                        channels=['P'])
        U_back = self.backward_data.load_data(lower_points, frame='geographic', coords='spherical', in_deg=False,
                                        channels=['UR','UT','UZ'])
        P_back = self.backward_data.load_data(lower_points, frame='geographic', coords='spherical', in_deg=False,
                                        channels=['P'])

        # Flip adjoint values
        U_back = np.flip(U_back, axis=2)
        P_back = np.flip(P_back, axis=2)

        # Interpolate onto the master_time
        interp_U = np.empty(U.shape[:-1] + (len(self.master_time),))
        interp_P = np.empty(P.shape[:-1] + (len(self.master_time),))
        interp_U_back = np.empty(U_back.shape[:-1] + (len(self.master_time),))
        interp_P_back = np.empty(P_back.shape[:-1] + (len(self.master_time),))

        for i in range(len(points)):
            interp_P[i,0] = np.interp(self.master_time, self.fw_time, P[i,0])
            interp_P_back[i,0] = np.interp(self.master_time, self.bw_time, P_back[i,0])
            for j in range(3):
                interp_U[i,j] = np.interp(self.master_time, self.fw_time, U[i,j])
                interp_U_back[i,j] = np.interp(self.master_time, self.bw_time, U_back[i,j])

        # Compute K_rho_0 in fluid
        # Compute time at the derivative points
        master_time_dt = self.master_time[1] - self.master_time[0]
        # Compute time derivatives wrt to time
        dU_dt = np.diff(interp_U, axis=2) / master_time_dt
        dU_back_dt = np.diff(interp_U_back, axis=2) / master_time_dt
        # make dot product
        fw_bw = np.sum(dU_dt * dU_back_dt, axis=1)
        K_rho_0 = rho_lower * integrate.simpson(fw_bw, dx=master_time_dt)

        return K_rho_0

    def evaluate_SS(self, points: np.ndarray, radius: float) -> np.ndarray:
        # Get limit points
        upper_points, lower_points = self._form_limit_points(points, radius)

        # Find rho_upper and lower (assuming radius is in decreasing order)
        radius_index = self.forward_data.base_model['DATA']['radius'].index(radius)
        rho_upper, rho_lower = np.array(self.forward_data.base_model['DATA']['rho'])[[radius_index, radius_index + 1]]

        # COMPUTE IN SOLID

        # Load U, U', T, T'zz T'tz T'zr, Grr Gtt Gzz Gzr Gzt, G'rr G'tt G'rt G'tr G'rz G'tz
        U = self.forward_data.load_data(upper_points, frame='geographic', coords='spherical', in_deg=False,
                                        channels=['UR','UT','UZ'])
        T = self.forward_data.load_data(upper_points, frame='geographic', coords='spherical', in_deg=False,
                                        channels=['SRR','STT','SZZ','STZ','SZR','SRT'])
        G = self.forward_data.load_data(upper_points, frame='geographic', coords='spherical', in_deg=False,
                                        channels=['GRR','GTT','GZZ','GZR','GZT'])
        U_back = self.backward_data.load_data(upper_points, frame='geographic', coords='spherical', in_deg=False,
                                        channels=['UR','UT','UZ'])
        T_back = self.backward_data.load_data(upper_points, frame='geographic', coords='spherical', in_deg=False,
                                        channels=['SZZ','STZ','SZR'])
        G_back = self.backward_data.load_data(upper_points, frame='geographic', coords='spherical', in_deg=False,
                                        channels=['GRR','GTT','GRT','GTR','GRZ','GTZ'])

        # Flip adjoint values
        U_back = np.flip(U_back, axis=2)
        T_back = np.flip(T_back, axis=2)
        G_back = np.flip(G_back, axis=2)

        # Interpolate onto the master_time
        interp_U = np.empty(U.shape[:-1] + (len(self.master_time),))
        interp_T = np.empty(T.shape[:-1] + (len(self.master_time),))
        interp_G = np.empty(G.shape[:-1] + (len(self.master_time),))
        interp_U_back = np.empty(U_back.shape[:-1] + (len(self.master_time),))
        interp_T_back = np.empty(T_back.shape[:-1] + (len(self.master_time),))
        interp_G_back = np.empty(G_back.shape[:-1] + (len(self.master_time),))

        for i in range(len(points)):
            for j in range(3):
                interp_U[i,j] = np.interp(self.master_time, self.fw_time, U[i,j])
                interp_U_back[i,j] = np.interp(self.master_time, self.bw_time, U_back[i,j])
                interp_T_back[i,j] = np.interp(self.master_time, self.bw_time, T_back[i,j])
            for j in range(5):
                interp_G[i,j] = np.interp(self.master_time, self.fw_time, G[i,j])
            for j in range(6):
                interp_T[i,j] = np.interp(self.master_time, self.fw_time, T[i,j])
                interp_G_back[i,j] = np.interp(self.master_time, self.bw_time, G_back[i,j])

        # Compute K_rho_0 in solid
        # Compute time at the derivative points
        master_time_dt = self.master_time[1] - self.master_time[0]
        # Compute time derivatives wrt to time
        dU_dt = np.diff(interp_U, axis=2) / master_time_dt
        dU_back_dt = np.diff(interp_U_back, axis=2) / master_time_dt
        # make dot product
        fw_bw = np.sum(dU_dt * dU_back_dt, axis=1)
        K_rho_0_upper = rho_upper * integrate.simpson(fw_bw, dx=master_time_dt)

        # Compute rest of Kdv + K_dn in solid
        integrand = interp_T[:,0,:] * interp_G_back[:,0,:] + interp_T[:,1,:] * interp_G_back[:,1,:] + \
                interp_T[:,5,:] * (interp_G_back[:,2,:] + interp_G[:,3,:]) + \
                interp_T[:,4,:] * interp_G_back[:,4,:] + interp_T[:,3,:] * interp_G_back[:,5,:] - \
                interp_T_back[:,2,:] * interp_G[:,3,:] - interp_T_back[:,1,:] * interp_G[:,4,:] - interp_T_back[:,0,:] * interp_G[:,2,:]

        K_dv_dn_upper = integrate.simpson(integrand, dx=master_time_dt)

        # Load U, U', T, T'zz T'tz T'zr, Grr Gtt Gzz Gzr Gzt, G'rr G'tt G'rt G'tr G'rz G'tz
        U = self.forward_data.load_data(lower_points, frame='geographic', coords='spherical', in_deg=False,
                                        channels=['UR','UT','UZ'])
        T = self.forward_data.load_data(lower_points, frame='geographic', coords='spherical', in_deg=False,
                                        channels=['SRR','STT','SZZ','STZ','SZR','SRT'])
        G = self.forward_data.load_data(lower_points, frame='geographic', coords='spherical', in_deg=False,
                                        channels=['GRR','GTT','GZZ','GZR','GZT'])
        U_back = self.backward_data.load_data(lower_points, frame='geographic', coords='spherical', in_deg=False,
                                        channels=['UR','UT','UZ'])
        T_back = self.backward_data.load_data(lower_points, frame='geographic', coords='spherical', in_deg=False,
                                        channels=['SZZ','STZ','SZR'])
        G_back = self.backward_data.load_data(lower_points, frame='geographic', coords='spherical', in_deg=False,
                                        channels=['GRR','GTT','GRT','GTR','GRZ','GTZ'])

        # Flip adjoint values
        U_back = np.flip(U_back, axis=2)
        T_back = np.flip(T_back, axis=2)
        G_back = np.flip(G_back, axis=2)

        # Interpolate onto the master_time
        interp_U = np.empty(U.shape[:-1] + (len(self.master_time),))
        interp_T = np.empty(T.shape[:-1] + (len(self.master_time),))
        interp_G = np.empty(G.shape[:-1] + (len(self.master_time),))
        interp_U_back = np.empty(U_back.shape[:-1] + (len(self.master_time),))
        interp_T_back = np.empty(T_back.shape[:-1] + (len(self.master_time),))
        interp_G_back = np.empty(G_back.shape[:-1] + (len(self.master_time),))

        for i in range(len(points)):
            for j in range(3):
                interp_U[i,j] = np.interp(self.master_time, self.fw_time, U[i,j])
                interp_U_back[i,j] = np.interp(self.master_time, self.bw_time, U_back[i,j])
                interp_T_back[i,j] = np.interp(self.master_time, self.bw_time, T_back[i,j])
            for j in range(5):
                interp_G[i,j] = np.interp(self.master_time, self.fw_time, G[i,j])
            for j in range(6):
                interp_T[i,j] = np.interp(self.master_time, self.fw_time, T[i,j])
                interp_G_back[i,j] = np.interp(self.master_time, self.bw_time, G_back[i,j])

        # Compute K_rho_0 in solid
        # Compute time at the derivative points
        master_time_dt = self.master_time[1] - self.master_time[0]
        # Compute time derivatives wrt to time
        dU_dt = np.diff(interp_U, axis=2) / master_time_dt
        dU_back_dt = np.diff(interp_U_back, axis=2) / master_time_dt
        # make dot product
        fw_bw = np.sum(dU_dt * dU_back_dt, axis=1)
        K_rho_0_lower = rho_lower * integrate.simpson(fw_bw, dx=master_time_dt)

         # Compute rest of Kdv + K_dn in solid
        integrand = interp_T[:,0,:] * interp_G_back[:,0,:] + interp_T[:,1,:] * interp_G_back[:,1,:] + \
                interp_T[:,5,:] * (interp_G_back[:,2,:] + interp_G[:,3,:]) + \
                interp_T[:,4,:] * interp_G_back[:,4,:] + interp_T[:,3,:] * interp_G_back[:,5,:] - \
                interp_T_back[:,2,:] * interp_G[:,3,:] - interp_T_back[:,1,:] * interp_G[:,4,:] - interp_T_back[:,0,:] * interp_G[:,2,:]

        K_dv_dn_lower = integrate.simpson(integrand, dx=master_time_dt)

        return K_rho_0_upper + K_dv_dn_upper - K_rho_0_lower - K_dv_dn_lower


    def evaluate_K_dv(self, points: np.ndarray, radius: float) -> np.ndarray:

        # Find the type of the discontinuity
        disc_type = self._find_discontinuity_type(radius)

        # Get limit points
        upper_points, lower_points = self._form_limit_points(points, radius)

        # Find rho_upper and lower (assuming radius is in decreasing order)
        radius_index = self.forward_data.base_model['DATA']['radius'].index(radius)
        rho_upper, rho_lower = np.array(self.forward_data.base_model['DATA']['rho'])[[radius_index, radius_index + 1]]
        vs_upper, vs_lower = np.array(self.forward_data.base_model['DATA']['vs'])[[radius_index, radius_index + 1]]
        vp_upper, vp_lower = np.array(self.forward_data.base_model['DATA']['vp'])[[radius_index, radius_index + 1]]

        # Compute the volumetric-geometric kernel
        if disc_type == 'SS':
            K_dv_upper = rho_upper * self.evaluate_rho_0(points=upper_points) + \
                    rho_upper * (vp_upper**2 - 2*vs_upper**2) * self.evaluate_lambda(points=upper_points) + \
                    rho_upper * vs_upper**2 * self.evaluate_mu(points=upper_points)
            K_dv_lower = rho_lower * self.evaluate_rho_0(points=lower_points) + \
                    rho_lower * (vp_lower**2 - 2*vs_lower**2) * self.evaluate_lambda(points=lower_points) + \
                    rho_lower * vs_lower**2 * self.evaluate_mu(points=lower_points)
        elif disc_type == 'FS':
            # wrong
            K_dv_upper = rho_upper * self.evaluate_rho_0(points=upper_points) + \
                    rho_upper * (vp_upper**2 - 2*vs_upper**2) * self.evaluate_lambda(points=upper_points) + \
                    rho_upper * vs_upper**2 * self.evaluate_mu(points=upper_points)
            K_dv_lower = rho_lower * self.evaluate_rho_0(points=lower_points) + \
                    rho_lower * vp_lower**2 * self.evaluate_lambda(points=lower_points)
        elif disc_type == 'SF':
            # wrong
            K_dv_upper = rho_upper * self.evaluate_rho_0(points=upper_points) + \
                    6 * rho_upper * vp_upper**2 * self.evaluate_lambda(points=upper_points)
            K_dv_lower = rho_lower * self.evaluate_rho_0(points=lower_points) + \
                    3 * rho_lower * (vs_lower**2 + 2*vp_lower**2) * self.evaluate_lambda(points=lower_points) + \
                    3 * rho_lower * vs_lower**2 * self.evaluate_mu(points=lower_points)

        return K_dv_upper - K_dv_lower


    def evaluate_K_dn(self, points: np.ndarray, radius: float) -> np.ndarray:
        # Get limit points
        upper_points, lower_points = self._form_limit_points(points, radius)

        # get forwards and backward waveforms at these points an flip andjoints in time
        Gr_forward_upper = np.nan_to_num(self.forward_data.load_data(upper_points,
                                        channels=['GZR', 'GZZ', 'GZT'], in_deg=False))
        Gr_backward_upper = np.nan_to_num(self.backward_data.load_data(upper_points,
                                        channels=['GZR', 'GZZ', 'GZT'], in_deg=False))
        Tr_forward_upper = np.nan_to_num(self.forward_data.load_data(upper_points,
                                        channels=['SZR', 'STZ', 'SZZ'], in_deg=False))
        Tr_backward_upper = np.nan_to_num(self.backward_data.load_data(upper_points,
                                        channels=['SZR', 'STZ', 'SZZ'], in_deg=False))
        Gr_forward_lower = np.nan_to_num(self.forward_data.load_data(lower_points,
                                        channels=['GZR', 'GZT', 'GZZ'], in_deg=False))
        Gr_backward_lower = np.nan_to_num(self.backward_data.load_data(lower_points,
                                        channels=['GZR', 'GZZ', 'GZT'], in_deg=False))
        Tr_forward_lower = np.nan_to_num(self.forward_data.load_data(lower_points,
                                        channels=['SZR', 'STZ', 'SZZ'], in_deg=False))
        Tr_backward_lower = np.nan_to_num(self.backward_data.load_data(lower_points,
                                        channels=['SZR', 'STZ', 'SZZ'], in_deg=False))

        # flip adjoints in time
        Gr_backward_upper = np.flip(Gr_backward_upper, axis=2)
        Gr_backward_lower = np.flip(Gr_backward_lower, axis=2)
        Tr_backward_upper = np.flip(Tr_backward_upper, axis=2)
        Tr_backward_lower = np.flip(Tr_backward_lower, axis=2)

        # Project on master time
        Gr_forward_upper_interp = np.empty(Gr_forward_upper.shape[:-1] + (len(self.master_time),))
        Gr_backward_upper_interp = np.empty(Gr_backward_upper.shape[:-1] + (len(self.master_time),))
        Tr_forward_upper_interp = np.empty(Tr_forward_upper.shape[:-1] + (len(self.master_time),))
        Tr_backward_upper_interp = np.empty(Tr_backward_upper.shape[:-1] + (len(self.master_time),))
        Gr_forward_lower_interp = np.empty(Gr_forward_lower.shape[:-1] + (len(self.master_time),))
        Gr_backward_lower_interp = np.empty(Gr_backward_lower.shape[:-1] + (len(self.master_time),))
        Tr_forward_lower_interp = np.empty(Tr_forward_lower.shape[:-1] + (len(self.master_time),))
        Tr_backward_lower_interp = np.empty(Tr_backward_lower.shape[:-1] + (len(self.master_time),))

        for i in range(len(points)):
            for j in range(3):
                Gr_forward_upper_interp[i,j] = np.interp(self.master_time, self.fw_time, Gr_forward_upper[i,j])
                Gr_backward_upper_interp[i,j] = np.interp(self.master_time, self.bw_time, Gr_backward_upper[i,j])
                Tr_forward_upper_interp[i,j] = np.interp(self.master_time, self.fw_time, Tr_forward_upper[i,j])
                Tr_backward_upper_interp[i,j] = np.interp(self.master_time, self.bw_time, Tr_backward_upper[i,j])
                Gr_forward_lower_interp[i,j] = np.interp(self.master_time, self.fw_time, Gr_forward_lower[i,j])
                Gr_backward_lower_interp[i,j] = np.interp(self.master_time, self.bw_time, Gr_backward_lower[i,j])
                Tr_forward_lower_interp[i,j] = np.interp(self.master_time, self.fw_time, Tr_forward_lower[i,j])
                Tr_backward_lower_interp[i,j] = np.interp(self.master_time, self.bw_time, Tr_backward_lower[i,j])

        # Compute the integrand
        integrand = np.sum(Tr_forward_upper_interp * Gr_backward_upper_interp, axis=1) + \
                    np.sum(Tr_backward_upper_interp * Gr_forward_upper_interp, axis=1) - \
                    np.sum(Tr_forward_lower_interp * Gr_backward_lower_interp, axis=1) - \
                    np.sum(Tr_backward_lower_interp * Gr_forward_lower_interp, axis=1)

        return -integrate.simpson(integrand,
                                 dx = (self.master_time[1] - self.master_time[0]))


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
        data = self.evaluate_vs(mesh.points)
        np.savetxt('slice_kernel.csv', data, delimiter=',')

        #data = np.loadtxt('/home/adrian/PhD/AxiSEM3D/slice_kernel.csv', delimiter=' ')

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