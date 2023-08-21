from abc import ABC, abstractmethod
import numpy as np

from axisem3d_output.aux.coordinate_transforms import sph2cart, cart2sph

class Mesh(ABC):

    @abstractmethod
    def plot_mesh(self):
        pass


class SliceMesh(Mesh):
        
    def __init__(self, point1: np.ndarray, point2: np.ndarray, domains: list, resolution: int,
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
        # Form vectors for the two points (Earth frame)
        if coord_in == 'spherical':
            self.point1 = sph2cart(point1)
            self.point2 = sph2cart(point2)
        else:
            self.point1 = point1
            self.point2 = point2

        self.domains = domains
        self.resolution = resolution
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
                                filtered_slice_points.append(cart2sph(point))
                            else:
                                filtered_slice_points.append(point)
                            filtered_indices.append([index1, index2])
                            in_domains = True

        self.indices = np.array(filtered_indices)
        self.points = np.array(filtered_slice_points)

    def plot_mesh(self):
        return super().plot_mesh()
    

class SphereMesh(Mesh):

    def __init__(self, resolution: int) -> None:
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
