from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ..aux.coordinate_transforms import sph2cart, cart2sph
from ..aux.helper_functions import find_range

class Mesh(ABC):

    @abstractmethod
    def plot_mesh(self):
        pass


class SliceMesh(Mesh):

    def __init__(self, data_frame_path: str = None,
                 point1: np.ndarray = None,
                 point2: np.ndarray = None,
                 domains: list = None, resolution: int = None,
                coord_in: str='spherical', coord_out: str='spherical') -> None:
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
            self.point1 = metadata['point1'][0]
            self.point2 = metadata['point2'][0]
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

    def plot_on_mesh(self, data: list = None, log_plot: bool = False,
                     low_range: float = 0, high_range: float = 1):
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

        if log_plot is False:
            _, cbar_max = find_range(slice_matrix,
                                           percentage_min=0,
                                           percentage_max=1)
            cbar_max *= (high_range * high_range)
            cbar_min = -cbar_max
            plt.figure()
            contour = plt.contourf(self.inplane_DIM1, self.inplane_DIM2,
                                   np.nan_to_num(slice_matrix,),
                                   levels=np.linspace(cbar_min, cbar_max, 100),
                                   cmap='RdBu_r', extend='both')
        else:
            log_10_sensitivity = np.log10(np.abs(slice_matrix,))
            cbar_min, cbar_max = self.find_range(log_10_sensitivity,
                                                  percentage_min=low_range,
                                                  percentage_max=high_range)

            plt.figure()
            contour = plt.contourf(self.inplane_DIM1, self.inplane_DIM2,
                                   log_10_sensitivity,
                                   levels=np.linspace(cbar_min, cbar_max, 100),
                                   cmap='RdBu_r', extend='both')

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
        plt.show()

class SphereMesh(Mesh):

    def __init__(self, resolution: int) -> None:
        self.type = 'sphere'

        self.resolution = resolution
        self._compute_mesh()

    def _compute_mesh(self):
        # Output is in format (lat, lon) in radians
        points = []
        lats = np.linspace(-np.pi, np.pi, self.resolution)
        for lat in lats:
            lons = np.linspace(-np.pi, np.pi, int(self.resolution * np.sin(lat)))
            for lon in lons:
                points.append([lat, lon])

        self.points =  np.array(points)
