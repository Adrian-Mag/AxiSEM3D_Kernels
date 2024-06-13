from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ..aux.coordinate_transforms import sph2cart, cart2sph
from ..aux.helper_functions import find_range
from mayavi import mlab
from scipy.spatial import ConvexHull
from matplotlib import cm
from scipy import stats
from tvtk.util import ctf
import matplotlib

class Mesh(ABC):

    @abstractmethod
    def plot_mesh(self):
        pass


class SliceMesh(Mesh):

    def __init__(self, data_frame_path: str = None,
                 point1: np.ndarray = None,
                 point2: np.ndarray = None,
                 domains: list = None, resolution: int = None,
                 coord_in: str='spherical', coord_out: str='spherical',
                 degrees: bool=True) -> None:
        """
        Create a mesh for a slice of Earth within a specified radius range and
        resolution. At the bases there is a square 2D uniform mesh whose
        coordinates are saved in inplane_DIM1 and inplane_DIM2. Each point in
        the mesh has two coordinates in the 2D mesh and also two associated
        indices within the inplane_DIM1 and inplane_DIM2 matrices. Only the
        physical coordinates of the points within the desired domains are
        outputted alongside their associated indices within the inplane_DIMi
        matrices.

        Args:
            point1 (np.ndarray): The source location [radius, latitude,longitude] in radians.
            point2 (np.ndarray): The station location [radius, latitude, longitude] in radians.
            coord (str): 'spherical' or 'cartesian'

        Returns:
            None

        """
        self.type = 'slice'
        if data_frame_path is not None:
            data_frame = pd.read_hdf(data_frame_path, 'df')
            metadata = pd.read_hdf(data_frame_path, 'metadata')
            self.data = data_frame['data'].values
            self.coord_in = metadata['coord_in'][0]
            self.coord_out = metadata['coord_out'][0]
            self.point1 = metadata['point1']
            self.point2 = metadata['point2']
            self.domains = metadata['domains']
            self.resolution = metadata['resolution']
            self._compute_basis()
            self._compute_mesh(self.domains, self.resolution, self.coord_out)

        else:
            self.data = None
            self.coord_in = coord_in
            self.coord_out = coord_out
            # Form vectors for the two points (Earth frame)
            if coord_in == 'spherical':
                if degrees:
                    point1[1::] = np.deg2rad(point1[1::])
                    point2[1::] = np.deg2rad(point2[1::])
                    self.point1 = sph2cart(np.array(point1))
                    self.point2 = sph2cart(np.array(point2))
                else:
                    self.point1 = sph2cart(np.array(point1))
                    self.point2 = sph2cart(np.array(point2))
            else:
                self.point1 = np.array(point1)
                self.point2 = np.array(point2)

            self.domains = domains
            self.resolution = resolution
            self.points = None
            self.indices = None
            self._compute_basis()
            self._compute_mesh(domains, resolution, coord_out)

    def _compute_basis(self):
        # Do Gram-Schmidt orthogonalization to form slice basis (Earth frame)
        self.base1 = self.point1 / np.linalg.norm(self.point1)
        self.base2 = self.point2 - np.dot(self.point2, self.base1) * self.base1
        self.base2 /= np.linalg.norm(self.base2)
        # base1 will be along the index1 in the inplane_DIM matrices and base2
        # along index2

    def _compute_mesh(self, domains: list, resolution: int, coord_out: str='spherical'):
        # Find the limits of the union of domains
        domains = np.array(domains)
        R_max = np.max(domains[:,1])

        # Generate index mesh
        indices_dim1 = np.arange(resolution)
        indices_dim2 = np.arange(resolution)

        # Generate in-plane mesh
        inplane_dim1 = np.linspace(-R_max, R_max, resolution)
        inplane_dim2 = np.linspace(-R_max, R_max, resolution)
        self.inplane_DIM1, self.inplane_DIM2 = np.meshgrid(inplane_dim1, inplane_dim2, indexing='ij')
        radii = np.sqrt(self.inplane_DIM1*self.inplane_DIM1 + self.inplane_DIM2*self.inplane_DIM2)
        thetas = np.arctan2(self.inplane_DIM2, self.inplane_DIM1)

        # Generate slice mesh points
        filtered_indices = []
        filtered_slice_points = []
        for index1 in indices_dim1:
            for index2 in indices_dim2:
                in_domains = False
                for domain in domains:
                    if not in_domains:
                        R_min = domain[0]
                        R_max = domain[1]
                        theta_min = domain[2]
                        theta_max = domain[3]
                        if radii[index1, index2] < R_max and radii[index1, index2] > R_min \
                            and thetas[index1, index2] < theta_max and thetas[index1, index2] > theta_min:
                            point = inplane_dim1[index1] * self.base1 + inplane_dim2[index2] * self.base2  # Slice frame -> Earth frame
                            if coord_out == 'spherical':
                                filtered_slice_points.append(cart2sph(point).reshape((3,)))
                            else:
                                filtered_slice_points.append(point)
                            filtered_indices.append([index1, index2])
                            in_domains = True

        self.indices = np.array(filtered_indices)
        self.points = np.array(filtered_slice_points)

    def plot_mesh(self):
        pass

    def plot_on_mesh(self, data: list = None,
                     high_range: float = 1,
                     filename: str = None,
                     cbar_range: list = None):
        if data is None:
            if self.data is None:
                raise ValueError('No data to plot.')
            else:
                data = self.data

        # Create matrix that will be plotted
        slice_matrix = np.full((self.resolution, self.resolution),
                                fill_value=np.NaN)
        # Distribute the values in the matrix
        index = 0
        for [index1, index2], _ in zip(self.indices, self.points):
            slice_matrix[index1, index2] = data[index]
            index += 1
        if cbar_range is None:
            _, cbar_max = find_range(slice_matrix,
                                            percentage_min=0,
                                            percentage_max=1)
            cbar_max *= (high_range * high_range)
            cbar_min = -cbar_max
        else:
            cbar_min, cbar_max = cbar_range
        plt.figure()
        # Add a circle of radius 6371000
        earth_circle = plt.Circle((0, 0), 6371000, edgecolor='white',
                                  fill=True, facecolor='black', alpha=0.2)
        contour = plt.contourf(self.inplane_DIM1, self.inplane_DIM2,
                            np.nan_to_num(slice_matrix,),
                            levels=np.linspace(cbar_min, cbar_max, 100),
                            cmap='RdBu_r', extend='both')
        plt.gca().add_artist(earth_circle)

        plt.scatter(np.dot(self.point1, self.base1),
                    np.dot(self.point1, self.base2))
        plt.scatter(np.dot(self.point2, self.base1),
                    np.dot(self.point2, self.base2))
        cbar = plt.colorbar(contour)

        cbar_ticks = np.linspace(cbar_min, cbar_max, 5)  # Example tick values
        cbar_ticklabels = ["{:.2e}".format(cbar_tick) for
                           cbar_tick in cbar_ticks] # Example tick labels
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels(cbar_ticklabels)
        cbar.set_label('Intensity')

        plt.xticks([])  # Remove x ticks
        plt.yticks([])  # Remove y ticks
        plt.gca().axis('off')
        if filename is not None:
            plt.savefig(filename + '.png')
        plt.show()

    def save_data(self, filename: str, data):
        df, metadata = self._create_dataframe(data)
        df.to_hdf(filename + '.h5', key='df', mode='w')
        metadata.to_hdf(filename + '.h5', key='metadata', mode='a')

    def _create_dataframe(self, data):
        radius = [coord[0] for coord in self.points]
        lat = [coord[1] for coord in self.points]
        lon = [coord[2] for coord in self.points]
        index_0 = [index[0] for index in self.indices]
        index_1 = [index[1] for index in self.indices]

        df = pd.DataFrame({
            'data': list(data),
            'radius': radius,
            'lat': lat,
            'lon': lon,
            'index_0': index_0,
            'index_1': index_1
        })

        # Store metadata in a Series
        metadata = pd.Series({
            'mesh.type': self.type,
            'point1': self.point1,
            'point2': self.point2,
            'resolution': self.resolution,
            'domains': self.domains,
            'coord_in': self.coord_in,
            'coord_out': self.coord_out,
        })

        return df, metadata

def matplotlib_to_mayavi(cmap):
    lut = cmap(np.linspace(0, 1, 256)) * 255  # Scale to 255 for RGB
    return lut

class SphereMesh:
    def __init__(self, data_frame_path: str=None,
                  radius: float=1, n: int=1000,
                  domain=None, degrees=True):
        self.type = 'sphere'
        if data_frame_path is not None:
            data_frame = pd.read_hdf(data_frame_path, 'df')
            metadata = pd.read_hdf(data_frame_path, 'metadata')
            self.n = metadata['n']
            self.data = data_frame['data'].values
            self.radius = metadata['radius']
            self.domain = metadata['domain']
            self.points = self.fibonacci_sphere(self.n)
        else:
            self.n = n
            self.radius = radius
            self.domain = domain
            if domain is not None:
                self.domain = np.array(domain)
                if degrees:
                    self.domain = np.deg2rad(self.domain)
            self.points = self.fibonacci_sphere(n)

    def fibonacci_sphere(self, n=1):
        # Points are returned in [rad, lat, lon] format in radians!
        points = []
        phi = (np.sqrt(5.0) - 1.0) / 2.0  # golden ratio
        for i in range(-n, n):
            lat = np.arcsin(2*i/(2*n+1))
            lon = 2 * np.pi * i / phi
            # Compute the sine and cosine of the angle
            sin = np.sin(lon)
            cos = np.cos(lon)

            # Use arctan2 to get the angle in the correct range
            lon = np.arctan2(sin, cos)

            if self.domain is not None:
                if (lat > self.domain[0] and
                    lat < self.domain[1] and
                    lon > self.domain[2] and
                    lon < self.domain[3]):

                    points.append([lat, lon])
            else:
                points.append([lat, lon])

        return np.array(points)

    def plot(self):
        # Convert (lat, lon) to (x, y, z)
        x = self.radius * np.cos(self.points[:, 0]) * np.cos(self.points[:, 1])
        y = self.radius * np.cos(self.points[:, 0]) * np.sin(self.points[:, 1])
        z = self.radius * np.sin(self.points[:, 0])

        # Create a new figure
        mlab.figure(size=(600, 600))

        # Plot the points
        mlab.points3d(x, y, z, color=(0, 0, 1), scale_factor=self.radius/50)

        # Show the figure
        mlab.show()

    def plot_on_mesh(self, data: list = None, gamma: float = 1.1,
                     remove_outliers: bool = True, cbar_range: list = None):
        if data is None:
            if self.data is None:
                raise ValueError('No data to plot.')
            else:
                data = self.data

        # Convert (lat, lon) to (x, y, z)
        x = self.radius * np.cos(self.points[:, 0]) * np.cos(self.points[:, 1])
        y = self.radius * np.cos(self.points[:, 0]) * np.sin(self.points[:, 1])
        z = self.radius * np.sin(self.points[:, 0])

        # Combine the x, y, z coordinates into a single array
        points = np.stack((x, y, z), axis=-1)

        # Compute the convex hull of the points
        hull = ConvexHull(points)

        triangles = hull.simplices

        # Use the simplices attribute of the result to get the triangles
        if cbar_range is None:
            if remove_outliers:
                z_scores = np.abs(stats.zscore(data))

                # Threshold is set to 3 standard deviations
                outliers = z_scores > 3

                # Get only the values that are not considered outliers
                data_no_outliers = data[~outliers]
                # Calculate the color values based on the interpolated_kernel data
                max_range = np.abs(data_no_outliers).max() * gamma
            else:
                max_range = np.abs(data).max() * gamma
                vmin, vmax = max_range, -max_range
        else:
            vmin, vmax = cbar_range

        cmap = matplotlib.cm.get_cmap('RdBu_r')
        mlab.figure(bgcolor=(1, 1, 1))
        my_triangular_mesh = mlab.triangular_mesh(x, y, z, triangles, scalars=data, vmin=vmin, vmax=vmax)
        my_triangular_mesh.module_manager.scalar_lut_manager.lut.table = matplotlib_to_mayavi(cmap)
        mlab.show()

    def save_data(self, filename: str, data):
        df, metadata = self._create_dataframe(data)
        df.to_hdf(filename + '.h5', key='df', mode='w')
        metadata.to_hdf(filename + '.h5', key='metadata', mode='a')

    def _create_dataframe(self, data):
        lat = [coord[0] for coord in self.points]
        lon = [coord[1] for coord in self.points]

        df = pd.DataFrame({
            'data': list(data),
            'lat': lat,
            'lon': lon,
        })

        # Store metadata in a Series
        metadata = pd.Series({
            'n': self.n,
            'mesh.type': self.type,
            'radius': self.radius,
            'domain': self.domain,
        })

        return df, metadata