from ..handlers.element_output import ElementOutput
from ...aux.helper_functions import window_data
from ...aux.mesher import Mesh, SliceMesh, SphereMesh
from ...aux.coordinate_transforms import sph2cart_mpmath, sph2cart, cart2sph, cart2sph_mpmath, cart2polar, cart_geo2cart_src, cart2cyl
import numpy as np
import pandas as pd
from scipy import integrate
import matplotlib
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
    def __init__(self, forward_obj: ElementOutput,
                 backward_obj: ElementOutput):
        # We store the forward and backward data
        self.forward_data = forward_obj
        self.backward_data = backward_obj

        self._compute_times()

        # Dictionary for kernel types:
        self.kernel_types = {'rho_0': self.evaluate_rho_0,
                             'lambda': self.evaluate_lambda,
                             'mu': self.evaluate_mu,
                             'rho': self.evaluate_rho,
                             'vp': self.evaluate_vp,
                             'vs': self.evaluate_vs,
                             'dV': self.evaluate_K_dv
                             }

    def _compute_times(self):
        # get the forward and backward time (assuming that all element groups
        # have the same time axis!!)
        first_group = self.forward_data.element_groups[0]
        self.fw_time = self.forward_data.element_groups_info[first_group]['metadata']['data_time'] # noqa
        self.fw_dt = self.fw_time[1] - self.fw_time[0]
        bw_time = self.backward_data.element_groups_info[first_group]['metadata']['data_time'] # noqa
        # Apply again t -> T-t transform on the adjoint time
        self.bw_time = np.flip(np.max(bw_time) - bw_time)
        self.bw_dt = self.bw_time[1] - self.bw_time[0]

        # Find the master time (minmax/maxmin)
        t_max = min(self.fw_time[-1], self.bw_time[-1])
        t_min = max(self.fw_time[0], self.bw_time[0])
        dt = max(self.fw_dt, self.bw_dt)
        self.master_time = np.arange(t_min, t_max + dt, dt)

    def evaluate_rho_0(self, points: np.ndarray) -> np.ndarray:
        # get forwards and backward displacements at these points
        forward_waveform = np.nan_to_num(
            self.forward_data.load_data(points=points,
                                        channels=['UR', 'UT', 'UZ'],
                                        in_deg=False)
            )
        backward_waveform = np.nan_to_num(
            self.backward_data.load_data(points=points,
                                         channels=['UR', 'UT', 'UZ'],
                                         in_deg=False))

        # Apply again t -> T-t transform on the adjoint data
        backward_waveform = np.flip(backward_waveform, axis=2)
        # Compute time at the derivative points
        fw_time = self.fw_time[0:-1] + self.fw_dt / 2
        bw_time = self.bw_time[0:-1] + self.bw_dt / 2
        # Compute derivatives wrt to time
        dfwdt = np.diff(forward_waveform, axis=2) / self.fw_dt
        dbwdt = np.diff(backward_waveform, axis=2) / self.bw_dt

        # Interpolate onto the master_time
        interp_dfwdt = np.empty(dfwdt.shape[:-1] + (len(self.master_time),))
        interp_dbwdt = np.empty(dbwdt.shape[:-1] + (len(self.master_time),))

        for i in range(dfwdt.shape[0]):
            for j in range(3):
                interp_dfwdt[i, j] = np.interp(self.master_time,
                                               fw_time, dfwdt[i, j])
                interp_dbwdt[i, j] = np.interp(self.master_time,
                                               bw_time, dbwdt[i, j])

        # make dot product
        fw_bw = np.sum(interp_dfwdt * interp_dbwdt, axis=1)

        sensitivity = - integrate.simpson(fw_bw, dx=(self.master_time[1] -
                                                   self.master_time[0]))
        return sensitivity

    def evaluate_lambda(self, points: np.ndarray) -> np.ndarray:
        # K_lambda^zero = int_T (div u)(div u^t) = int_T (tr E)(tr E^t) =
        # int_T (GZZ+GRR+GTT)(GZZ^t+GRR^t+GTT^t)
        material_mapping = self.forward_data._group_by_material(points)
        solid_points = points[material_mapping==0]
        liquid_points = points[material_mapping==1]
        sensitivity = np.zeros(len(points))

        if len(solid_points) > 0:
            # Compute solid sensitivities
            # get forwards and backward waveforms at this point
            forward_diagG = np.nan_to_num(
                self.forward_data.load_data(solid_points,
                                            channels=['GZZ', 'GRR', 'GTT'],
                                            in_deg=False))
            backward_diagG = np.nan_to_num(
                self.backward_data.load_data(solid_points,
                                            channels=['GZZ', 'GRR', 'GTT'],
                                            in_deg=False))

            # compute trace of each wavefield and flip adjoint in time
            trace_G = forward_diagG.sum(axis=1)
            trace_G_adjoint = np.flip(backward_diagG.sum(axis=1), axis=1)

            # Project both on master time
            interp_trace_G = np.empty(trace_G.shape[:-1] +
                                    (len(self.master_time),))
            interp_trace_G_adjoint = np.empty(trace_G.shape[:-1] +
                                            (len(self.master_time),))

            for i in range(len(solid_points)):
                interp_trace_G[i] = np.interp(self.master_time,
                                            self.fw_time,
                                            trace_G[i])
                interp_trace_G_adjoint[i] = np.interp(self.master_time,
                                                    self.bw_time,
                                                    trace_G_adjoint[i])
            dt = self.master_time[1] - self.master_time[0]
            solid_sensitivities = integrate.simpson(interp_trace_G * interp_trace_G_adjoint,
                                    dx=dt) # noqa
            sensitivity[material_mapping==0] = solid_sensitivities

        if len(liquid_points) > 0:
        # Compute liquid sensitivities
            forward_P = np.nan_to_num(
                self.forward_data.load_data(liquid_points,
                                            channels=['P'],
                                            in_deg=False))[:,0,:]
            backward_P = np.nan_to_num(
                self.backward_data.load_data(liquid_points,
                                            channels=['P'],
                                            in_deg=False))[:,0,:]
            # flip adjpoint in time
            backward_P = np.flip(backward_P, axis=1)

            interp_forward_P = np.empty(forward_P.shape[:-1] +
                                        (len(self.master_time),))
            interp_backward_P = np.empty(backward_P.shape[:-1] +
                                        (len(self.master_time),))

            for i in range(len(liquid_points)):
                interp_forward_P[i] = np.interp(self.master_time,
                                                self.fw_time,
                                                forward_P[i])
                interp_backward_P[i] = np.interp(self.master_time,
                                                self.bw_time,
                                                backward_P[i])
            # Get material properties
            rho = self._find_material_property(liquid_points, 'rho')
            vp = self._find_material_property(liquid_points, 'vp')

            factor = 1 / ((rho**2) * vp**4)
            dt = self.master_time[1] - self.master_time[0]
            liquid_sensitivities = integrate.simpson(interp_forward_P * interp_backward_P * factor[:, np.newaxis],
                                                    dx=dt)
            sensitivity[material_mapping==1] = liquid_sensitivities

        return sensitivity

    def evaluate_mu(self, points: np.ndarray) -> np.ndarray:
        # K_mu_0 = int_T (grad u^t):(grad u) + (grad u^t):(grad u)^T
        # = int_T 2E^t:E

        # We first try to compute the sensitivity using the strain tensor, but
        # if it is not available, then we will use the gradient of displacement
        material_mapping = self.forward_data._group_by_material(points)
        solid_points = points[material_mapping==0]
        liquid_points = points[material_mapping==1]
        sensitivity = np.zeros(len(points))
        if len(solid_points) > 0:
        # get forwards and backward waveforms at this point
            G_forward = np.nan_to_num(
                self.forward_data.load_data(
                    solid_points, channels=['GRR', 'GRT', 'GRZ',
                                    'GTR', 'GTT', 'GTZ',
                                    'GZR', 'GZT', 'GZZ'], in_deg=False))
            G_adjoint = np.nan_to_num(
                self.backward_data.load_data(
                    solid_points, channels=['GRR', 'GRT', 'GRZ',
                                    'GTR', 'GTT', 'GTZ',
                                    'GZR', 'GZT', 'GZZ'], in_deg=False))

            # flip adjoint in time
            G_adjoint = np.flip(G_adjoint, axis=2)

            # Project both arrays on the master time
            interp_G_forward = np.empty(G_forward.shape[:-1] +
                                        (len(self.master_time),))
            interp_G_adjoint = np.empty(G_adjoint.shape[:-1] +
                                        (len(self.master_time),))
            for i in range(len(solid_points)):
                for j in range(9):
                    interp_G_forward[i, j, :] = np.interp(self.master_time,
                                                        self.fw_time,
                                                        G_forward[i, j, :])
                    interp_G_adjoint[i, j, :] = np.interp(self.master_time,
                                                        self.bw_time,
                                                        G_adjoint[i, j, :])

            interp_G_forward = interp_G_forward.reshape(len(solid_points), 3, 3,
                                                        len(self.master_time))
            interp_G_adjoint = interp_G_adjoint.reshape(len(solid_points), 3, 3,
                                                        len(self.master_time))

            # Multiply
            integrand = np.sum(interp_G_adjoint *
                            (interp_G_forward +
                                interp_G_forward.transpose(0, 2, 1, 3)),
                            axis=(1, 2))
            dt = self.master_time[1] - self.master_time[0]
            solid_sensitivities = integrate.simpson(integrand, dx=dt)
            sensitivity[material_mapping==0] = solid_sensitivities

        if len(liquid_points) > 0:
        # Compute liquid sensitivities
            forward_P = np.nan_to_num(
                self.forward_data.load_data(liquid_points,
                                            channels=['P'],
                                            in_deg=False))[:,0,:]
            backward_P = np.nan_to_num(
                self.backward_data.load_data(liquid_points,
                                            channels=['P'],
                                            in_deg=False))[:,0,:]
            # flip adjpoint in time
            backward_P = np.flip(backward_P, axis=1)

            interp_forward_P = np.empty(forward_P.shape[:-1] +
                                        (len(self.master_time),))
            interp_backward_P = np.empty(backward_P.shape[:-1] +
                                        (len(self.master_time),))

            for i in range(len(liquid_points)):
                interp_forward_P[i] = np.interp(self.master_time,
                                                self.fw_time,
                                                forward_P[i])
                interp_backward_P[i] = np.interp(self.master_time,
                                                self.bw_time,
                                                backward_P[i])
            # Get material properties
            rho = self._find_material_property(liquid_points, 'rho')
            vp = self._find_material_property(liquid_points, 'vp')

            factor = 2 / (3 * (rho**2) * vp**4)
            liquid_sensitivities = integrate.simpson(interp_forward_P * interp_backward_P * factor[:, np.newaxis],
                                                    dx=dt)
            sensitivity[material_mapping==1] = liquid_sensitivities

        return sensitivity

    def evaluate_rho(self, points: np.ndarray) -> np.ndarray:
        # K_rho = K_rho_0 + (vp^2-2vs^2)K_lambda_0 + vs^2 K_mu_0

        vp = self.forward_data._load_material_property(points, 'vp')
        vs = self.forward_data._load_material_property(points, 'vs')

        result = self.evaluate_rho_0(points) + \
            (vp * vp - 2 * vs * vs) * self.evaluate_lambda(points) + \
            vs * vs * self.evaluate_mu(points)
        return result

    def evaluate_vp(self, points):
        # K_vs = 2*rho*vp*K_lambda_0
        vp = self.forward_data._load_material_property(points, 'vp')
        rho = self.forward_data._load_material_property(points, 'rho')

        return 2 * rho * vp * self.evaluate_lambda(points)

    def evaluate_vs(self, points):
        # K_vs = 2*rho*vs*(K_mu_0 - 2*K_lambda_0)
        vs = self.forward_data._load_material_property(points, 'vs')
        rho = self.forward_data._load_material_property(points, 'rho')

        result = 2 * rho * vs * (self.evaluate_mu(points) - 2 *
                                 self.evaluate_lambda(points))
        return result

    def evaluate_geometric(self, points: np.ndarray, radius: float):
        # All points must be in the format  (lat, lon) in radians, where lat
        # and lon are in the geographical frame. The radius must be in meters.
        # This only works for solid-solid now
        pass

    def _find_discontinuity_type(self, radius: float):
        # Find the desired discontinuity in the base model (for 1D only)
        if radius in self.forward_data.base_model['DISCONTINUITIES']:
            # Find what type of discontinuity it is (only SS works for now)
            radius_index = self.forward_data.base_model['DATA']['radius'].index(radius) # noqa
            [vs_upper, vs_lower] = np.array(
                self.forward_data.base_model['DATA']['vs']
                )[[radius_index, radius_index+1]]

            if vs_upper > 0 and vs_lower > 0:
                return 'SS'
            elif vs_upper > 0 and vs_lower == 0:
                return 'FS'
            elif vs_upper == 0 and vs_lower > 0:
                return 'SF'
        else:
            raise ValueError('There is no discontinuity at {}. \
                             The available discontinuities are at {}'.format(radius, # noqa
                                                                             self.forward_data.base_model['DISCONTINUITIES']) # noqa
                             )

    def _form_limit_points(self, points: np.ndarray,
                           radius: float) -> np.ndarray:
        # Form upper and lower limit points
        dr = 1000
        upper_points = np.array([[radius + dr, lat, lon] for
                                 lat, lon in points])
        lower_points = np.array([[radius - dr, lat, lon] for
                                 lat, lon in points])

        return (upper_points, lower_points)

    def evaluate_on_sphere_2(self, n: int, radius: float, parameter: str):
        mesh = SphereMesh(n, radius)
        # Compute sensitivity values on the slice (Slice frame)
        data = self.kernel_types[parameter](mesh.points, radius)

        return mesh, data

    def evaluate_on_sphere(self, resolution: int, radius: int):
        # Compute points on spherical mesh
        # Define the lat lon grid (must match data file)
        lat = np.arange(-45, 45.01, 1)*np.pi/180
        lon = np.arange(-40, 120.01, 1)*np.pi/180
        LON, LAT = np.meshgrid(lon, lat)
        nlat = len(lat)
        nlon = len(lon)
        points = np.dstack((LAT, LON)).reshape(-1, 2)

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

        kernel = np.loadtxt('/home/adrian/PhD/AxiSEM3D/moho.csv',
                            delimiter=' ')

        kernel = kernel.reshape(LON.shape)

        # Define the lat lon grid for the larger mesh covering the Earth's
        # surface
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
        N = len(interpolated_kernel.flatten())  # Number of points
        # Key point: set an integer for each point
        scalars = np.arange(N).reshape(interpolated_kernel.shape[0],
                                       interpolated_kernel.shape[1])

        # Choose a colormap from matplotlib (e.g., 'viridis', 'plasma',
        # 'cividis', etc.)
        cmap = cm.get_cmap('RdBu_r')  # Change this to your desired colormap

        # Calculate the color values based on the interpolated_kernel data
        gamma = 1.1
        max_range = interpolated_kernel.max() ** gamma
        color_values = (interpolated_kernel + max_range) / (2*max_range)

        # Convert color values to RGBA using the chosen colormap
        colors = (cmap(color_values) * 255).astype(np.uint8)

        # Set alpha channel to 255 (no transparency)
        colors[:, :, 3] = 255
        # Reshape the colors array to be 2D to (N, 4) where N is the total
        # number of points
        colors_reshaped = colors.reshape(-1, 4)

        mlab.figure(bgcolor=(0, 0, 0))
        # Plot interpolated_kernel
        interpolated_kernel_surface = mlab.mesh(X_disc, Y_disc, Z_disc,
                                                scalars=scalars, mode='sphere',
                                                opacity=1)
        # Set look-up table and redraw
        interpolated_kernel_surface.module_manager.scalar_lut_manager.lut.table = colors_reshaped # noqa

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

    def evaluate_CMB_solid(self, points: np.ndarray,
                           radius: float) -> np.ndarray:
        # Get limit points
        upper_points, lower_points = self._form_limit_points(points, radius)

        # Find rho_upper and lower (assuming radius is in decreasing order)
        radius_index = self.forward_data.base_model['DATA']['radius'].index(radius) # noqa
        rho_upper, rho_lower = np.array(
            self.forward_data.base_model['DATA']['rho']
            )[[radius_index, radius_index + 1]]

        # COMPUTE IN SOLID

        # Load U, U', T, T'zz T'tz T'zr, Grr Gtt Gzz Gzr Gzt, G'rr G'tt G'rt
        # G'tr G'rz G'tz
        U = self.forward_data.load_data(upper_points, frame='geographic',
                                        coords='spherical', in_deg=False,
                                        channels=['UR', 'UT', 'UZ'])
        T = self.forward_data.load_data(upper_points, frame='geographic',
                                        coords='spherical', in_deg=False,
                                        channels=['SRR', 'STT', 'SZZ',
                                                  'STZ', 'SZR', 'SRT'])
        G = self.forward_data.load_data(upper_points, frame='geographic',
                                        coords='spherical', in_deg=False,
                                        channels=['GRR', 'GTT', 'GZZ',
                                                  'GZR', 'GZT'])
        U_back = self.backward_data.load_data(upper_points, frame='geographic',
                                              coords='spherical', in_deg=False,
                                              channels=['UR', 'UT', 'UZ'])
        T_back = self.backward_data.load_data(upper_points, frame='geographic',
                                              coords='spherical', in_deg=False,
                                              channels=['SZZ', 'STZ', 'SZR'])
        G_back = self.backward_data.load_data(upper_points, frame='geographic',
                                              coords='spherical', in_deg=False,
                                              channels=['GRR', 'GTT', 'GRT',
                                                        'GTR', 'GRZ', 'GTZ'])

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
                interp_U[i, j] = np.interp(self.master_time, self.fw_time,
                                           U[i, j])
                interp_U_back[i, j] = np.interp(self.master_time, self.bw_time,
                                                U_back[i, j])
                interp_T_back[i, j] = np.interp(self.master_time, self.bw_time,
                                                T_back[i, j])
            for j in range(5):
                interp_G[i, j] = np.interp(self.master_time, self.fw_time,
                                           G[i, j])
            for j in range(6):
                interp_T[i, j] = np.interp(self.master_time, self.fw_time,
                                           T[i, j])
                interp_G_back[i, j] = np.interp(self.master_time, self.bw_time,
                                                G_back[i, j])

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
        integrand = interp_T[:, 0, :] * interp_G_back[:, 0, :] + \
            interp_T[:, 1, :] * interp_G_back[:, 1, :] + \
            interp_T[:, 5, :] * (interp_G_back[:, 2, :] +
                                 interp_G[:, 3, :]) + \
            interp_T[:, 4, :] * interp_G_back[:, 4, :] + \
            interp_T[:, 3, :] * interp_G_back[:, 5, :] - \
            interp_T_back[:, 2, :] * interp_G[:, 3, :] - \
            interp_T_back[:, 1, :] * interp_G[:, 4, :] - \
            interp_T_back[:, 0, :] * interp_G[:, 2, :]

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

        return K_rho_0 + K_dv_dn  # + K_d_sigma

    def evaluate_CMB_fluid(self, points: np.ndarray,
                           radius: float) -> np.ndarray:
        # Get limit points
        upper_points, lower_points = self._form_limit_points(points, radius)

        # Find rho_upper and lower (assuming radius is in decreasing order)
        radius_index = self.forward_data.base_model['DATA']['radius'].index(radius) # noqa
        rho_upper, rho_lower = np.array(
            self.forward_data.base_model['DATA']['rho']
            )[[radius_index, radius_index + 1]]

        # COMPUTE IN FLUID

        # Load U, U', P, P', Grr Gtt Gzz
        U = self.forward_data.load_data(lower_points, frame='geographic',
                                        coords='spherical', in_deg=False,
                                        channels=['UR', 'UT', 'UZ'])
        P = self.forward_data.load_data(lower_points, frame='geographic',
                                        coords='spherical', in_deg=False,
                                        channels=['P'])
        U_back = self.backward_data.load_data(lower_points, frame='geographic',
                                              coords='spherical', in_deg=False,
                                              channels=['UR', 'UT', 'UZ'])
        P_back = self.backward_data.load_data(lower_points, frame='geographic',
                                              coords='spherical', in_deg=False,
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
            interp_P[i, 0] = np.interp(self.master_time, self.fw_time,
                                       P[i, 0])
            interp_P_back[i, 0] = np.interp(self.master_time, self.bw_time,
                                            P_back[i, 0])
            for j in range(3):
                interp_U[i, j] = np.interp(self.master_time, self.fw_time,
                                           U[i, j])
                interp_U_back[i, j] = np.interp(self.master_time, self.bw_time,
                                                U_back[i, j])

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
        radius_index = self.forward_data.base_model['DATA']['radius'].index(radius) # noqa
        rho_upper, rho_lower = np.array(
            self.forward_data.base_model['DATA']['rho']
            )[[radius_index, radius_index + 1]]

        # COMPUTE IN SOLID

        # Load U, U', T, T'zz T'tz T'zr, Grr Gtt Gzz Gzr Gzt, G'rr G'tt G'rt
        # G'tr G'rz G'tz
        U = self.forward_data.load_data(upper_points, frame='geographic',
                                        coords='spherical', in_deg=False,
                                        channels=['UR', 'UT', 'UZ'])
        T = self.forward_data.load_data(upper_points, frame='geographic',
                                        coords='spherical', in_deg=False,
                                        channels=['SRR', 'STT', 'SZZ',
                                                  'STZ', 'SZR', 'SRT'])
        G = self.forward_data.load_data(upper_points, frame='geographic',
                                        coords='spherical', in_deg=False,
                                        channels=['GRR', 'GTT', 'GZZ',
                                                  'GZR', 'GZT'])
        U_back = self.backward_data.load_data(upper_points, frame='geographic',
                                              coords='spherical', in_deg=False,
                                              channels=['UR', 'UT', 'UZ'])
        T_back = self.backward_data.load_data(upper_points, frame='geographic',
                                              coords='spherical', in_deg=False,
                                              channels=['SZZ', 'STZ', 'SZR'])
        G_back = self.backward_data.load_data(upper_points, frame='geographic',
                                              coords='spherical', in_deg=False,
                                              channels=['GRR', 'GTT', 'GRT',
                                                        'GTR', 'GRZ', 'GTZ'])

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
                interp_U[i, j] = np.interp(self.master_time, self.fw_time,
                                           U[i, j])
                interp_U_back[i, j] = np.interp(self.master_time, self.bw_time,
                                                U_back[i, j])
                interp_T_back[i, j] = np.interp(self.master_time, self.bw_time,
                                                T_back[i, j])
            for j in range(5):
                interp_G[i, j] = np.interp(self.master_time, self.fw_time,
                                           G[i, j])
            for j in range(6):
                interp_T[i, j] = np.interp(self.master_time, self.fw_time,
                                           T[i, j])
                interp_G_back[i, j] = np.interp(self.master_time, self.bw_time,
                                                G_back[i, j])

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
        integrand = interp_T[:, 0, :] * interp_G_back[:, 0, :] + \
            interp_T[:, 1, :] * interp_G_back[:, 1, :] + \
            interp_T[:, 5, :] * (interp_G_back[:, 2, :] +
                                 interp_G[:, 3, :]) + \
            interp_T[:, 4, :] * interp_G_back[:, 4, :] + \
            interp_T[:, 3, :] * interp_G_back[:, 5, :] - \
            interp_T_back[:, 2, :] * interp_G[:, 3, :] - \
            interp_T_back[:, 1, :] * interp_G[:, 4, :] - \
            interp_T_back[:, 0, :] * interp_G[:, 2, :]

        K_dv_dn_upper = integrate.simpson(integrand, dx=master_time_dt)

        # Load U, U', T, T'zz T'tz T'zr, Grr Gtt Gzz Gzr Gzt, G'rr G'tt G'rt
        # G'tr G'rz G'tz
        U = self.forward_data.load_data(lower_points, frame='geographic',
                                        coords='spherical', in_deg=False,
                                        channels=['UR', 'UT', 'UZ'])
        T = self.forward_data.load_data(lower_points, frame='geographic',
                                        coords='spherical', in_deg=False,
                                        channels=['SRR', 'STT', 'SZZ',
                                                  'STZ', 'SZR', 'SRT'])
        G = self.forward_data.load_data(lower_points, frame='geographic',
                                        coords='spherical', in_deg=False,
                                        channels=['GRR', 'GTT', 'GZZ',
                                                  'GZR', 'GZT'])
        U_back = self.backward_data.load_data(lower_points, frame='geographic',
                                              coords='spherical', in_deg=False,
                                              channels=['UR', 'UT', 'UZ'])
        T_back = self.backward_data.load_data(lower_points, frame='geographic',
                                              coords='spherical', in_deg=False,
                                              channels=['SZZ', 'STZ', 'SZR'])
        G_back = self.backward_data.load_data(lower_points, frame='geographic',
                                              coords='spherical', in_deg=False,
                                              channels=['GRR', 'GTT', 'GRT',
                                                        'GTR', 'GRZ', 'GTZ'])

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
                interp_U[i, j] = np.interp(self.master_time, self.fw_time,
                                           U[i, j])
                interp_U_back[i, j] = np.interp(self.master_time, self.bw_time,
                                                U_back[i, j])
                interp_T_back[i, j] = np.interp(self.master_time, self.bw_time,
                                                T_back[i, j])
            for j in range(5):
                interp_G[i, j] = np.interp(self.master_time, self.fw_time,
                                           G[i, j])
            for j in range(6):
                interp_T[i, j] = np.interp(self.master_time, self.fw_time,
                                           T[i, j])
                interp_G_back[i, j] = np.interp(self.master_time, self.bw_time,
                                                G_back[i, j])

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
        integrand = interp_T[:, 0, :] * interp_G_back[:, 0, :] + \
            interp_T[:, 1, :] * interp_G_back[:, 1, :] + \
            interp_T[:, 5, :] * (interp_G_back[:, 2, :] +
                                 interp_G[:, 3, :]) + \
            interp_T[:, 4, :] * interp_G_back[:, 4, :] + \
            interp_T[:, 3, :] * interp_G_back[:, 5, :] - \
            interp_T_back[:, 2, :] * interp_G[:, 3, :] - \
            interp_T_back[:, 1, :] * interp_G[:, 4, :] - \
            interp_T_back[:, 0, :] * interp_G[:, 2, :]

        K_dv_dn_lower = integrate.simpson(integrand, dx=master_time_dt)

        return K_rho_0_upper + K_dv_dn_upper - K_rho_0_lower - K_dv_dn_lower

    def evaluate_Kd(self, points: np.ndarray, radius: float) -> np.ndarray:
        return self.evaluate_K_dn(points, radius) + self.evaluate_K_dv(points, radius)

    def evaluate_K_dv(self, points: np.ndarray, radius: float) -> np.ndarray:

        # Find the type of the discontinuity
        disc_type = self._find_discontinuity_type(radius)

        # Get limit points
        upper_points, lower_points = self._form_limit_points(points, radius)

        # Find properties above and below the discontinuity (assuming radius is
        # in decreasing order)
        rho_lower = self.forward_data._load_material_property(lower_points, 'rho')
        rho_upper = self.forward_data._load_material_property(upper_points, 'rho')
        vp_lower = self.forward_data._load_material_property(lower_points, 'vp')
        vp_upper = self.forward_data._load_material_property(upper_points, 'vp')
        vs_lower = self.forward_data._load_material_property(lower_points, 'vs')
        vs_upper = self.forward_data._load_material_property(upper_points, 'vs')

        # Compute the volumetric-geometric kernel (upper/lower)
        if disc_type == 'SS':
            K_dv_upper = rho_upper * \
                self.evaluate_rho_0(points=upper_points) + \
                rho_upper * (vp_upper**2 - 2*vs_upper**2) * \
                self.evaluate_lambda(points=upper_points) + \
                rho_upper * vs_upper**2 * self.evaluate_mu(points=upper_points)
            K_dv_lower = rho_lower * \
                self.evaluate_rho_0(points=lower_points) + \
                rho_lower * (vp_lower**2 - 2*vs_lower**2) * \
                self.evaluate_lambda(points=lower_points) + \
                rho_lower * vs_lower**2 * self.evaluate_mu(points=lower_points)
        elif disc_type == 'FS':
            K_dv_upper = rho_upper * \
                self.evaluate_rho_0(points=upper_points) + \
                rho_upper * (vp_upper**2 - 2*vs_upper**2) * \
                self.evaluate_lambda(points=upper_points) + \
                rho_upper * vs_upper**2 * self.evaluate_mu(points=upper_points)
            K_dv_lower = rho_lower * \
                self.evaluate_rho_0(points=lower_points) + \
                rho_lower * vp_lower**2 * \
                self.evaluate_lambda(points=lower_points)
        elif disc_type == 'SF':
            K_dv_upper = rho_upper * \
                self.evaluate_rho_0(points=upper_points) + \
                6 * rho_upper * vp_upper**2 * \
                self.evaluate_lambda(points=upper_points)
            K_dv_lower = rho_lower * \
                self.evaluate_rho_0(points=lower_points) + \
                3 * rho_lower * (vs_lower**2 + 2*vp_lower**2) *\
                self.evaluate_lambda(points=lower_points) + \
                3 * rho_lower * vs_lower**2 * \
                self.evaluate_mu(points=lower_points)

        return K_dv_lower - K_dv_upper

    def evaluate_K_dn(self, points: np.ndarray, radius: float) -> np.ndarray:
        # Get limit points
        upper_points, lower_points = self._form_limit_points(points, radius)

        # Find the type of the discontinuity
        disc_type = self._find_discontinuity_type(radius)

        # get forwards and backward waveforms at these points an flip andjoints
        # in time

        if disc_type == 'SS':
            # Load forward and backward waveforms
            Gr_forward_upper = np.nan_to_num(
                self.forward_data.load_data(upper_points,
                                            channels=['GZR', 'GZZ', 'GZT'],
                                            in_deg=False))
            Gr_backward_upper = np.nan_to_num(
                self.backward_data.load_data(upper_points,
                                            channels=['GZR', 'GZZ', 'GZT'],
                                            in_deg=False))
            Tr_forward_upper = np.nan_to_num(
                self.forward_data.load_data(upper_points,
                                            channels=['SZR', 'STZ', 'SZZ'],
                                            in_deg=False))
            Tr_backward_upper = np.nan_to_num(
                self.backward_data.load_data(upper_points,
                                            channels=['SZR', 'STZ', 'SZZ'],
                                            in_deg=False))
            Gr_forward_lower = np.nan_to_num(
            self.forward_data.load_data(lower_points,
                                        channels=['GZR', 'GZT', 'GZZ'],
                                        in_deg=False))
            Gr_backward_lower = np.nan_to_num(
                self.backward_data.load_data(lower_points,
                                            channels=['GZR', 'GZZ', 'GZT'],
                                            in_deg=False))
            Tr_forward_lower = np.nan_to_num(
                self.forward_data.load_data(lower_points,
                                            channels=['SZR', 'STZ', 'SZZ'],
                                            in_deg=False))
            Tr_backward_lower = np.nan_to_num(
                self.backward_data.load_data(lower_points,
                                            channels=['SZR', 'STZ', 'SZZ'],
                                            in_deg=False))
            # flip adjoints in time
            Gr_backward_upper = np.flip(Gr_backward_upper, axis=2)
            Gr_backward_lower = np.flip(Gr_backward_lower, axis=2)
            Tr_backward_upper = np.flip(Tr_backward_upper, axis=2)
            Tr_backward_lower = np.flip(Tr_backward_lower, axis=2)
            # Project on master time
            Gr_forward_upper_interp = np.empty(Gr_forward_upper.shape[:-1] +
                                            (len(self.master_time),))
            Gr_backward_upper_interp = np.empty(Gr_backward_upper.shape[:-1] +
                                                (len(self.master_time),))
            Tr_forward_upper_interp = np.empty(Tr_forward_upper.shape[:-1] +
                                            (len(self.master_time),))
            Tr_backward_upper_interp = np.empty(Tr_backward_upper.shape[:-1] +
                                                (len(self.master_time),))
            Gr_forward_lower_interp = np.empty(Gr_forward_lower.shape[:-1] +
                                            (len(self.master_time),))
            Gr_backward_lower_interp = np.empty(Gr_backward_lower.shape[:-1] +
                                                (len(self.master_time),))
            Tr_forward_lower_interp = np.empty(Tr_forward_lower.shape[:-1] +
                                            (len(self.master_time),))
            Tr_backward_lower_interp = np.empty(Tr_backward_lower.shape[:-1] +
                                                (len(self.master_time),))
            for i in range(len(points)):
                for j in range(3):
                    Gr_forward_upper_interp[i, j] = np.interp(self.master_time,
                                                            self.fw_time,
                                                            Gr_forward_upper[i, j]) # noqa
                    Gr_backward_upper_interp[i, j] = np.interp(self.master_time,
                                                            self.bw_time,
                                                            Gr_backward_upper[i, j]) # noqa
                    Tr_forward_upper_interp[i, j] = np.interp(self.master_time,
                                                            self.fw_time,
                                                            Tr_forward_upper[i, j]) # noqa
                    Tr_backward_upper_interp[i, j] = np.interp(self.master_time,
                                                            self.bw_time,
                                                            Tr_backward_upper[i, j]) # noqa
                    Gr_forward_lower_interp[i, j] = np.interp(self.master_time,
                                                            self.fw_time,
                                                            Gr_forward_lower[i, j]) # noqa
                    Gr_backward_lower_interp[i, j] = np.interp(self.master_time,
                                                            self.bw_time,
                                                            Gr_backward_lower[i, j]) # noqa
                    Tr_forward_lower_interp[i, j] = np.interp(self.master_time,
                                                            self.fw_time,
                                                            Tr_forward_lower[i, j]) # noqa
                    Tr_backward_lower_interp[i, j] = np.interp(self.master_time,
                                                            self.bw_time,
                                                            Tr_backward_lower[i, j]) # noqa

            # Compute the integrand
            integrand = np.sum(Tr_forward_upper_interp * Gr_backward_upper_interp,
                            axis=1) + \
                np.sum(Tr_backward_upper_interp * Gr_forward_upper_interp,
                    axis=1) - \
                np.sum(Tr_forward_lower_interp * Gr_backward_lower_interp,
                    axis=1) - \
                np.sum(Tr_backward_lower_interp * Gr_forward_lower_interp,
                    axis=1)

            return -integrate.simpson(
                integrand, dx=(self.master_time[1] - self.master_time[0])
                )
        elif disc_type == 'FS':
            P_forward_upper = np.nan_to_num(
                self.forward_data.load_data(upper_points,
                                            channels=['P'],
                                            in_deg=False))
            P_backward_upper = np.nan_to_num(
                self.backward_data.load_data(upper_points,
                                            channels=['P'],
                                            in_deg=False))
            Gr_forward_lower = np.nan_to_num(
            self.forward_data.load_data(lower_points,
                                        channels=['GZR', 'GZT', 'GZZ'],
                                        in_deg=False))
            Gr_backward_lower = np.nan_to_num(
                self.backward_data.load_data(lower_points,
                                            channels=['GZR', 'GZZ', 'GZT'],
                                            in_deg=False))
            Tr_forward_lower = np.nan_to_num(
                self.forward_data.load_data(lower_points,
                                            channels=['SZR', 'STZ', 'SZZ'],
                                            in_deg=False))
            Tr_backward_lower = np.nan_to_num(
                self.backward_data.load_data(lower_points,
                                            channels=['SZR', 'STZ', 'SZZ'],
                                            in_deg=False))
            # flip adjoints in time
            P_backward_upper = np.flip(P_backward_upper, axis=2)
            Gr_backward_lower = np.flip(Gr_backward_lower, axis=2)
            Tr_backward_lower = np.flip(Tr_backward_lower, axis=2)
            # Project on master time
            P_forward_upper_interp = np.empty(P_forward_upper.shape[:-1] +
                                            (len(self.master_time),))
            P_backward_upper_interp = np.empty(P_backward_upper.shape[:-1] +
                                                (len(self.master_time),))
            Gr_forward_lower_interp = np.empty(Gr_forward_lower.shape[:-1] +
                                            (len(self.master_time),))
            Gr_backward_lower_interp = np.empty(Gr_backward_lower.shape[:-1] +
                                                (len(self.master_time),))
            Tr_forward_lower_interp = np.empty(Tr_forward_lower.shape[:-1] +
                                            (len(self.master_time),))
            Tr_backward_lower_interp = np.empty(Tr_backward_lower.shape[:-1] +
                                                (len(self.master_time),))
            for i in range(len(points)):
                for j in range(3):
                    P_forward_upper_interp[i, j] = np.interp(self.master_time,
                                                            self.fw_time,
                                                            Gr_forward_upper[i, j]) # noqa
                    P_backward_upper_interp[i, j] = np.interp(self.master_time,
                                                            self.bw_time,
                                                            Gr_backward_upper[i, j]) # noqa
                    Gr_forward_lower_interp[i, j] = np.interp(self.master_time,
                                                            self.fw_time,
                                                            Gr_forward_lower[i, j]) # noqa
                    Gr_backward_lower_interp[i, j] = np.interp(self.master_time,
                                                            self.bw_time,
                                                            Gr_backward_lower[i, j]) # noqa
                    Tr_forward_lower_interp[i, j] = np.interp(self.master_time,
                                                            self.fw_time,
                                                            Tr_forward_lower[i, j]) # noqa
                    Tr_backward_lower_interp[i, j] = np.interp(self.master_time,
                                                            self.bw_time,
                                                            Tr_backward_lower[i, j]) # noqa
            # Find properties above and below the discontinuity (assuming radius is
            # in decreasing order)
            rho_lower = self.forward_data._load_material_property(lower_points, 'rho')
            rho_upper = self.forward_data._load_material_property(upper_points, 'rho')
            vp_lower = self.forward_data._load_material_property(lower_points, 'vp')
            vp_upper = self.forward_data._load_material_property(upper_points, 'vp')
            vs_lower = self.forward_data._load_material_property(lower_points, 'vs')
            vs_upper = self.forward_data._load_material_property(upper_points, 'vs')
            factor = (2 - 1.5*(vs_upper/vp_upper)**2) / (3 * rho_upper * vp_upper**2)
            # Compute the integrand
            integrand = factor * np.sum(P_forward_upper_interp * P_backward_upper_interp,
                            axis=1) + \
                np.sum(Tr_forward_lower_interp * Gr_backward_lower_interp,
                    axis=1) - \
                np.sum(Tr_backward_lower_interp * Gr_forward_lower_interp,
                    axis=1)

            return -integrate.simpson(
                integrand, dx=(self.master_time[1] - self.master_time[0])
                )
        elif disc_type == 'SF':
            P_forward_lower = np.nan_to_num(
                self.forward_data.load_data(lower_points,
                                            channels=['P'],
                                            in_deg=False))
            P_backward_lower = np.nan_to_num(
                self.backward_data.load_data(lower_points,
                                            channels=['P'],
                                            in_deg=False))
            Gr_forward_upper = np.nan_to_num(
            self.forward_data.load_data(upper_points,
                                        channels=['GZR', 'GZT', 'GZZ'],
                                        in_deg=False))
            Gr_backward_upper = np.nan_to_num(
                self.backward_data.load_data(upper_points,
                                            channels=['GZR', 'GZZ', 'GZT'],
                                            in_deg=False))
            Tr_forward_upper = np.nan_to_num(
                self.forward_data.load_data(upper_points,
                                            channels=['SZR', 'STZ', 'SZZ'],
                                            in_deg=False))
            Tr_backward_upper = np.nan_to_num(
                self.backward_data.load_data(upper_points,
                                            channels=['SZR', 'STZ', 'SZZ'],
                                            in_deg=False))
            # flip adjoints in time
            P_backward_lower = np.flip(P_backward_lower, axis=2)
            Gr_backward_upper = np.flip(Gr_backward_upper, axis=2)
            Tr_backward_upper = np.flip(Tr_backward_upper, axis=2)
            # Project on master time
            P_forward_lower_interp = np.empty(P_forward_lower.shape[:-1] +
                                            (len(self.master_time),))
            P_backward_lower_interp = np.empty(P_backward_lower.shape[:-1] +
                                                (len(self.master_time),))
            Gr_forward_upper_interp = np.empty(Gr_forward_upper.shape[:-1] +
                                            (len(self.master_time),))
            Gr_backward_upper_interp = np.empty(Gr_backward_upper.shape[:-1] +
                                                (len(self.master_time),))
            Tr_forward_upper_interp = np.empty(Tr_forward_upper.shape[:-1] +
                                            (len(self.master_time),))
            Tr_backward_upper_interp = np.empty(Tr_backward_upper.shape[:-1] +
                                                (len(self.master_time),))
            for i in range(len(points)):
                for j in range(3):
                    P_forward_lower_interp[i, j] = np.interp(self.master_time,
                                                            self.fw_time,
                                                            Gr_forward_lower[i, j]) # noqa
                    P_backward_lower_interp[i, j] = np.interp(self.master_time,
                                                            self.bw_time,
                                                            Gr_backward_lower[i, j]) # noqa
                    Gr_forward_upper_interp[i, j] = np.interp(self.master_time,
                                                            self.fw_time,
                                                            Gr_forward_upper[i, j]) # noqa
                    Gr_backward_upper_interp[i, j] = np.interp(self.master_time,
                                                            self.bw_time,
                                                            Gr_backward_upper[i, j]) # noqa
                    Tr_forward_upper_interp[i, j] = np.interp(self.master_time,
                                                            self.fw_time,
                                                            Tr_forward_upper[i, j]) # noqa
                    Tr_backward_upper_interp[i, j] = np.interp(self.master_time,
                                                            self.bw_time,
                                                            Tr_backward_upper[i, j]) # noqa
            # Find properties above and below the discontinuity (assuming radius is
            # in decreasing order)
            rho_lower = self.forward_data._load_material_property(lower_points, 'rho')
            rho_upper = self.forward_data._load_material_property(upper_points, 'rho')
            vp_lower = self.forward_data._load_material_property(lower_points, 'vp')
            vp_upper = self.forward_data._load_material_property(upper_points, 'vp')
            vs_lower = self.forward_data._load_material_property(lower_points, 'vs')
            vs_upper = self.forward_data._load_material_property(upper_points, 'vs')
            factor = (2 - 1.5*(vs_lower/vp_lower)**2) / (3 * rho_lower * vp_lower**2)
            # Compute the integrand
            integrand = factor * np.sum(P_forward_lower_interp * P_backward_lower_interp,
                            axis=1) + \
                np.sum(Tr_forward_upper_interp * Gr_backward_upper_interp,
                    axis=1) - \
                np.sum(Tr_backward_upper_interp * Gr_forward_upper_interp,
                    axis=1)

            return -integrate.simpson(
                integrand, dx=(self.master_time[1] - self.master_time[0])
                )


    def evaluate_on_slice(self, parameter: str,
                          source_location: list = None,
                          station_location: list = None,
                          resolution: int = 50, domains: list = None,
                          log_plot: bool = False, low_range: float = 0.1,
                          high_range: float = 0.999, save_data: bool = True,
                          filename: str = 'slice_kernel',
                          plot_data: bool = True):
        # We first create a slice mesh, which needs point1, point2, domains and
        # resolution

        # Create default domains if None were given
        if domains is None:
            domains = []
            for element_group in self.forward_data.element_groups_info.values(): # noqa
                domains.append(element_group['elements']['vertical_range'] +
                               [-2*np.pi, 2*np.pi])
        domains = np.array(domains)
        # Create point1 and point2 if none were given
        R_max = np.max(domains[:, 1])
        if source_location is None and station_location is None:
            source_location = np.array([R_max,
                                        np.radians(
                                            self.forward_data.source_lat),
                                        np.radians(
                                            self.forward_data.source_lon)])
            station_location = np.array([R_max,
                                         np.radians(
                                             self.backward_data.source_lat),
                                         np.radians(
                                             self.backward_data.source_lon)])
        else:
            source_location = np.array(source_location)
            station_location = np.array(station_location)

        # Create a slice mesh
        mesh = SliceMesh(point1=source_location,
                         point2=station_location,
                         domains=domains,
                         resolution=resolution)

        # Compute sensitivity values on the slice (Slice frame)
        data = self.kernel_types[parameter](mesh.points)
        # Create a dataframe and metadata of the mesh + sensitivity
        data_frame, metadata = mesh._create_dataframe(data)
        # Save the infor and plot the sensitivity if desired
        if save_data:
            mesh.save_data(filename, data_frame, metadata)
        if plot_data:
            mesh.plot_on_mesh(data_frame['data'].values,
                              log_plot=log_plot,
                              low_range=low_range,
                              high_range=high_range)

        return data_frame

    def _point_in_region_of_interest(self, point) -> bool:
        if random.random() < 0.01:
            max_lat = 30
            min_lat = -30
            min_lon = -30
            max_lon = 30
            min_rad = 3400000
            max_rad = 6371000
            if point[0] < max_rad and point[0] > min_rad \
                    and np.rad2deg(point[1]) < max_lat \
                    and np.rad2deg(point[1]) > min_lat \
                    and np.rad2deg(point[2]) < max_lon \
                    and np.rad2deg(point[2]) > min_lon:
                return True
            else:
                return False
        else:
            return False

    def _find_material_property(self, points, material_property):
        # Find the indices within the basemodel where the points are located
        radii = np.array(self.forward_data.base_model['DATA']['radius'])
        is_increasing = radii[0] < radii[1]
        if is_increasing:
            index = np.searchsorted(radii, points[:, 0])
        else:
            index = np.searchsorted(-radii, -points[:, 0])
        # eliminated points outside of the domain
        mask = np.logical_or(index > 0, index < len(radii))
        filtered_index = index[mask]

        return np.array(
            self.forward_data.base_model['DATA'][material_property]
            )[filtered_index - 1]