import matplotlib 
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import numpy as np
import xarray as xr
import matplotlib 
matplotlib.use('tkagg')
import yaml
import pandas as pd
import xarray as xr
import obspy 
from obspy.core.inventory import Inventory, Network, Station, Channel
from tqdm import tqdm
import concurrent.futures
import time
import warnings 
import plotly.graph_objects as go
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import sys

from .axisem3d_output import AxiSEM3DOutput
from ...aux.coordinate_transforms import sph2cart, cart2sph, cart2polar, cart_geo2cart_src, cart2cyl
from ...aux.mesher import Mesh, SliceMesh

#warnings.filterwarnings("error")

class ElementOutput(AxiSEM3DOutput):
    def __init__(self, path_to_element_output:str) -> None:
        """Initializes the ElementOutput object for the given path to the element output directory.

        Args:
            path_to_element_output (str): Path to the element output directory (called "elements").

        Attributes:
            path_to_elements_output (str): Path to the element output directory.
            element_groups_info (Dict[str, Any]): Information about all the element groups.
            source_lat (float): Latitude of the event located on the axis.
            source_lon (float): Longitude of the event located on the axis.
            source_depth (float): Depth of the event located on the axis.
            rotation_matrix (np.ndarray): Rotation matrix for the wavefields.
        """
        path_to_simulation = self._find_simulation_path(path_to_element_output)
        super().__init__(path_to_simulation)
        self.path_to_elements_output = path_to_element_output

        # Load element groups information from the inparam output
        self.element_groups_info = self._load_element_groups_info()

        # Get lat lon of the event located on the axis
        self.source_lat, self.source_lon, self.source_depth = self._get_source_location()

        # Create the metadata for each element group
        self._create_element_metadata()

        # Compute rotation matrix
        self.rotation_matrix = self._compute_rotation_matrix()


    def _create_element_metadata(self) -> None:
        """Create metadata for each element group."""
        for element in self.element_groups_info:
            metadata = self._read_element_metadata(element)
            self.element_groups_info[element]['metadata'] = {
                'na_grid': metadata[0],
                'data_time': metadata[1],
                'list_element_na': metadata[2],
                'list_element_coords': metadata[3],
                'dict_list_element': metadata[4],
                'files': metadata[5],
                'elements_index_limits': metadata[6],
                'detailed_channels': metadata[7],
            }

            # Replace the numerical indicators of coordinates with letters based on the coordinate system
            coordinate_frame = self.element_groups_info[element]['wavefields']['coordinate_frame']
            self.element_groups_info[element]['metadata']['detailed_channels'] = [
                element.replace('1', coordinate_frame[0]).replace('2', coordinate_frame[1]).replace('3', coordinate_frame[2])
                for element in self.element_groups_info[element]['metadata']['detailed_channels']
            ]


    def _get_source_location(self) -> tuple[float, float, float]:
        """Get latitude, longitude, and depth of the event located on the axis."""
        with open(self.inparam_source, 'r') as file:
            source_yaml = yaml.load(file, Loader=yaml.FullLoader)
            source_name = list(source_yaml.get('list_of_sources', [{}])[0].keys())[0]
            # Assume a single point source
            source = source_yaml.get('list_of_sources', [{}])[0].get(source_name, {})
            source_lat, source_lon = source.get('location', {}).get('latitude_longitude', [0.0, 0.0])
            source_depth = float(source.get('location', {}).get('depth', 0.0))
        return source_lat, source_lon, source_depth
    

    def _load_element_groups_info(self) -> dict[str, any]:
        """Load element groups information from the inparam output."""
        element_groups_info = {}
        with open(self.inparam_output, 'r') as file:
            output_yaml = yaml.load(file, Loader=yaml.FullLoader)
            for dictionary in output_yaml.get('list_of_element_groups', []):
                element_group = next(iter(dictionary), None)
                if element_group:
                    element_groups_info[element_group] = dictionary[element_group]
        return element_groups_info


    def plot_mesh(self, special_elements):
        problematic_elements = []
        for index, element in enumerate(self.list_element_coords):
            s = element[:,0]
            z = element[:,1]
            points = cart2polar(s,z)
            r = points[[0,1,2],0]
            theta = points[[0,3,6],1]
            r_grid, theta_grid = np.meshgrid(r, theta)
            expected_points = np.column_stack((r_grid.ravel(), theta_grid.ravel()))
            if not np.allclose(points, expected_points, rtol=1e-6):
                problematic_elements.append(index)
        
        all_points = self.list_element_coords[problematic_elements].reshape(-1,2)
        unique_points = np.unique(all_points, axis=0)
        s_coords = unique_points[:, 0]
        z_coords = unique_points[:, 1]
        
        special_points = self.list_element_coords[special_elements].reshape(-1,2)
        unique_points = np.unique(special_points, axis=0)
        special_s_coords = unique_points[:,0]
        special_z_coords = unique_points[:,1]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=s_coords, y=z_coords, mode='markers', marker=dict(size=1, color='blue')))
        fig.add_trace(go.Scatter(x=special_s_coords, y=special_z_coords, mode='markers', marker=dict(size=1, color='red')))
        # Decrease the marker size
        marker_size = 5  # Change this value to adjust marker size
        fig.update_traces(marker=dict(size=marker_size))
        # Set aspect ratio to 1:1
        fig.update_layout(
            autosize=False,
            width=1600,
            height=1600,
        )
        fig.show()


    def obspyfy(self, path_to_station_file: str):
        # Create obspyfy folder if not existent already
        obspyfy_path = self.path_to_elements_output + '/obspyfied'
        if not os.path.exists(obspyfy_path):
            os.mkdir(obspyfy_path) 
        cat = self.catalogue
        cat.write(obspyfy_path + '/cat.xml', format='QUAKEML')

        stations_file_name = os.path.basename(path_to_station_file).split('.')[0]
        inv = self.create_inventory(path_to_station_file)
        inv.write(obspyfy_path + '/' + stations_file_name + '_inv.xml', format="stationxml")

        stream = self.stream_STA(path_to_station_file)
        stream.write(obspyfy_path + '/' + stations_file_name + '.mseed', format="MSEED") 


    def create_inventory(self, path_to_station_file: str):
        ##################
        # Create Inventory
        ##################

        networks = []
        station_names = []
        locations = []
        channels_list = []

        # Get path to where the new inventory will be saved, and coordinates

        # Create new empty inventory
        inv = Inventory(
            networks=[],
            source="Inventory from axisem STATIONS file")

        # Open station file
        stations = (pd.read_csv(path_to_station_file, 
                    delim_whitespace=True, 
                    header=0, 
                    names=["name","network","latitude","longitude","useless","depth"]))

        # Iterate over all stations in the stations file
        for _, station in stations.iterrows():
            # Create network if not already existent
            net_exists = False
            for network in inv:
                if network.code == station['network']:
                    net_exists = True
                    net = network
            if net_exists == False:
                net = Network(
                code=station['network'],
                stations=[])
                # add new network to inventory
                inv.networks.append(net)

            # Create station (should be unique!)
            sta = Station(
            code=station['name'],
            latitude=station['latitude'],
            longitude=station['longitude'],
            elevation=-station['depth'])
            net.stations.append(sta)

            # Create the channels
            # here we must find in which element group is this station located
            rad = self.Earth_Radius - station['depth']
            lat = np.deg2rad(station['latitude'])
            lon = np.deg2rad(station['longitude'])
            point = np.array([rad, lat, lon]).reshape(1,3)
            # get inplane coords
            point = sph2cart(point)
            point = cart2cyl(cart_geo2cart_src(points=point, 
                                                rotation_matrix=self.rotation_matrix))
            point = cart2polar(point[0,0], point[0,1])
            point[0,1] += np.pi/2
            element_group = self._separate_by_inplane_domain(point.reshape(1,2))
            key = list(self.element_groups_info.keys())[element_group[0]]
            for channel in self.element_groups_info[key]['metadata']['detailed_channels']:
                cha = Channel(
                code=channel,
                location_code="",
                latitude=station['latitude'],
                longitude=station['longitude'],
                elevation=-station['depth'],
                depth=station['depth'],
                azimuth=None,
                dip=None,
                sample_rate=None)
                sta.channels.append(cha)
            
            # Form the lists that will be used as inputs with read_netcdf
            # to get the stream of the wavefield data
            networks.append(station['network'])
            station_names.append(station['name'])
            locations.append('') # Axisem does not use locations
            channels_list.append(self.element_groups_info[key]['metadata']['detailed_channels'])

        return inv


    def stream_STA(self, path_to_station_file: str, 
                   channels: list = None,
                   time_limits: list = None) -> obspy.Stream:
        """Takes in the path to a station file used for axisem3d
        and returns a stream with the wavefields computed at all stations

        Args:
            path_to_station_file (str): path to station.txt file

        Returns:
            obspy.stream: stream
        """        
        # Get time slices from time limits
        if time_limits is not None:
            time_slices = np.where((self.data_time >= time_limits[0]) & (self.data_time <= time_limits[1]))
        else:
            time_slices = None

        # Open station file
        stations = (pd.read_csv(path_to_station_file, 
                    delim_whitespace=True, 
                    header=0, 
                    names=["name","network","latitude","longitude","useless","depth"]))
        # initiate stream that will hold data 
        stream = obspy.Stream()
        for _, station in stations.iterrows():
            stalat = station['latitude']
            stalon = station['longitude']
            stadepth = station['depth']
            starad = self.Earth_Radius - stadepth
            
            # Find the element group of this point
            point = np.array([starad, stalat, stalon]).reshape(1,3)
            point = sph2cart(point)
            point = cart2cyl(cart_geo2cart_src(points=point, 
                                                rotation_matrix=self.rotation_matrix))
            point = cart2polar(point[0,0], point[0,1])
            point[0,1] += np.pi/2
            element_group = self._separate_by_inplane_domain(point.reshape(1,2))
            key = list(self.element_groups_info.keys())[element_group[0]]

            # get the data at this station (assuming RTZ components)
            wave_data = self.load_data(points=np.array([starad, stalat, stalon]),
                                        channels=channels, 
                                        time_slices=time_slices)
            # Construct metadata
            data_time = self.element_groups_info[key]['metadata']['data_time']
            delta = data_time[1] - data_time[0]
            npts = len(data_time)
            network = station['network']
            station_name = station['name']
            if channels is not None:
                selected_detailed_channels = [element for element in self.element_groups_info[key]['metadata']['detailed_channels'] \
                                            if any(element.startswith(prefix) for prefix in channels)]
            else:
                selected_detailed_channels = self.element_groups_info[key]['metadata']['detailed_channels']
            for chn_index, chn in enumerate(selected_detailed_channels):
                # form the traces at the channel level
                trace = obspy.Trace(wave_data[0][chn_index])
                trace.stats.delta = delta
                trace.stats.ntps = npts
                trace.stats.network = network
                trace.stats.station = station_name
                trace.stats.location = ''
                trace.stats.channel = chn
                trace.stats.starttime = obspy.UTCDateTime("1970-01-01T00:00:00.0Z") + data_time[0]
                stream.append(trace)

        return stream


    def stream(self, points: np.ndarray, coord_in_deg: bool = True, 
               channels: list = None,
               time_limits: list = None) -> obspy.Stream:
        """
        Generate a stream with the wavefields computed at all stations given the location.

        Args:
            point (list): The location of the station in meters and radians/degrees.
                        It should be a list with the following elements:
                        - radial position in meters (float)
                        - latitude in radians/degrees (float)
                        - longitude in radians/degrees (float)
            coord_in_deg (bool, optional): Specify whether the coordinates are in degrees.
                                        Defaults to True.
            channels (list, optional): List of wavefield channels to include.
                                    Defaults to None, which includes all channels.
            time_limits (list, optional): Time limits for the data.
                                        It should be a list with two elements:
                                        - start time in seconds (float)
                                        - end time in seconds (float)
                                        Defaults to None, which includes all times.
            fourier_order (int, optional): Fourier order. Defaults to None.

        Returns:
            obspy.Stream: A stream containing the wavefields computed at all stations.
        """ 

        # Get time slices from time limits. We assume that all points have the same time axis!!!!
        first_key = next(iter(self.element_groups_info))
        data_time = self.element_groups_info[first_key]['metadata']['data_time']
        if time_limits is not None:
            time_slices = np.where((data_time >= time_limits[0]) & (data_time <= time_limits[1]))
        else:
            time_slices = None
        group = next(iter(self.element_groups_info))

        points = np.array(points)
        if points.ndim == 1:
            # In case we receive a point as a vector ... 
            points = points.reshape((1,3))

        # initiate stream that will hold data 
        stream = obspy.Stream()
        # get the data at this station (assuming RTZ components)
        wave_data = self.load_data(points=points,
                                    channels=channels, 
                                    time_slices=time_slices, 
                                    in_deg=coord_in_deg)
        for point_index, point in enumerate(points):
            # Construct metadata 
            delta = data_time[1] - data_time[0]
            npts = len(data_time)
            network = str(np.random.randint(0, 100))
            station_name = str(np.random.randint(0, 100))

            if channels is not None:
                    selected_detailed_channels = [element for element in self.element_groups_info[group]['metadata']['detailed_channels'] \
                                                if any(element.startswith(prefix) for prefix in channels)]
            else:
                selected_detailed_channels = self.element_groups_info[group]['metadata']['detailed_channels']
            for chn_index, chn in enumerate(selected_detailed_channels):
                # form the traces at the channel level
                trace = obspy.Trace(wave_data[point_index][chn_index])
                trace.stats.delta = delta
                trace.stats.ntps = npts
                trace.stats.network = network
                trace.stats.station = station_name
                trace.stats.location = ''
                trace.stats.channel = chn
                trace.stats.starttime = obspy.UTCDateTime("1970-01-01T00:00:00.0Z") + data_time[0]
                stream.append(trace)

        return stream


    def load_data(self, points: np.ndarray, frame: str='geographic', 
                  coords: str='spherical', in_deg: bool=True,
                  channels: list=None, time_slices: list=None):
        # Only options for coords are geographic+spherical,
        # geographic+cartesian, source+cylindrical.

        # If we receive only one point as ndarray([a,b,c]) we turn it into
        # ndarray([[a,b,c]])
        if points.ndim == 1:
            points = points.reshape((1,3))
        
        # Create channel slices and time slices if not given. Since all groups
        # are assumed to have the same time axis and to contain the necessary
        # channels, we will use here the first group
        group = next(iter(self.element_groups_info))
        if channels is None:
            channels = self.element_groups_info[group]['metadata']['detailed_channels']
        if time_slices is None:
            time_slices = np.arange(len(self.element_groups_info[next(iter(self.element_groups_info))]['metadata']['data_time']))
        channel_slices = self._channel_slices(channels, group)
        
        # Initialize the final result
        final_result = np.ones((len(points),len(channel_slices),len(time_slices)))
            
        # Make sure the data type of points is floats
        points = points.astype(np.float64)

        # Transforms points to cylindrical coords in source frame
        if frame == 'geographic':
            if coords == 'spherical':
                if in_deg is True:
                    points[:, 1:] = np.deg2rad(points[:, 1:])
                points = sph2cart(points)
            points = cart2cyl(cart_geo2cart_src(points=points, 
                                                rotation_matrix=self.rotation_matrix))
            logging.info("Transformed points to cylindrical coordinates in source frame.")

        # Compute the inplane coordinates

        # We add pi/2 beacause by default cart2polar sets theta to zero along
        # the s axis, but in the inparam.output theta is set to 0 at the z axis
        inplane_points = cart2polar(points[:,0], points[:,1])
        inplane_points[:,1] += np.pi/2

        # Separate inplane points by the element group they are part of
        group_mapping = self._separate_by_inplane_domain(inplane_points)    

        # Interpolate all groups
        for group_index, element_group in enumerate(self.element_groups_info):
            point_index = np.where(group_mapping==group_index)
            group_points = points[point_index]
            if len(group_points) != 0:
                final_result[point_index,:,:] = self.load_data_from_element_group(group_points, element_group,
                                                                                channel_slices, time_slices)

        return final_result


    def _separate_by_inplane_domain(self, inplane_points: np.ndarray, domains: list=None) -> np.ndarray:
        """
        Assigns each point in the given ndarray to one of the specified domains based on their polar coordinates.

        Args:
            inplane_points (numpy.ndarray): The ndarray containing points in polar coordinates [radius, theta].
            domains (list): List of domains specified as [r_min, r_max, theta_min, theta_max].

        Returns:
            numpy.ndarray: An array containing the assigned domain index for each point.
                        If a point does not fall within any domain, it is assigned the value -1.
        """
        if domains == None:
            # Then we use as domains the domains defined by the element groups:
            # Find all the domains available in the outputs
            domains = []
            for element_group in self.element_groups_info.values():
                r_min, r_max = element_group['elements']['vertical_range']
                theta_min, theta_max = element_group['elements']['horizontal_range']
                domains.append([r_min, r_max, theta_min, theta_max])

        # Initialize array to store assigned domains
        group_mapping = np.zeros(inplane_points.shape[0], dtype=int)  
        
        for i, point in enumerate(inplane_points):
            radius, theta = point
            # Loop through each domain and check if the point falls within it
            not_in_any_domain = True
            for domain_idx, domain in enumerate(domains):
                r_min, r_max, theta_min, theta_max = domain
                if r_min <= radius <= r_max and theta_min <= theta <= theta_max:
                    group_mapping[i] = domain_idx  # Assign domain index to the point
                    not_in_any_domain = False
                    break  # No need to check other domains
            if not_in_any_domain:
                group_mapping[i] = -1

        return np.array(group_mapping)
    

    def load_data_from_element_group(self, points: np.ndarray, group: str,
                                    channel_slices: list=None, time_slices: list=None):
        # All points must be from the same element group!!!

        # Initialize the final result
        final_result = np.ones((len(points),len(channel_slices),len(time_slices)))

        # Get lagrange order
        if self.element_groups_info[group]['inplane']['GLL_points_one_edge'] == [0,2,4] and \
            self.element_groups_info[group]['inplane']['edge_dimension'] == 'BOTH':
            lagrange_order = 3
            logging.info('Using lagrange order 3.')
        else:
            logging.error("No implementation for this output type.")
            raise NotImplementedError("No implementation for this output type.")

        # Find which points are unique in the inplane domain
        unique_points_dict = {}
        unique_points = []
        for name, point in enumerate(points):
            # Some points may have the same s-z coordinates but after
            # transformations the numerical errors made them different, so we
            # round the transformed points to decimals decimals
            decimals = 3
            s, z, _ = np.around(point, decimals=decimals)
            key = (s, z)
            if key in unique_points_dict:
                unique_points_dict[key][name] = point
            else:
                unique_points_dict[key] = {name: point}
                unique_points.append(point)
        unique_points = np.array(unique_points)
        logging.info('Only {}% of the points have unique inplane coords.'.format(100*len(unique_points)/len(points)))

        # Create element map for the unique points
        element_centers = self.element_groups_info[group]['metadata']['list_element_coords'][:, 4, :]  # Assuming the center point is at index 4
        differences = element_centers[:, np.newaxis] - unique_points[:,0:2]
        distances = np.linalg.norm(differences, axis=2)
        elements_map = np.argmin(distances, axis=0)
        logging.info('Created element map')

        # Create elements_dict
        elements_dict = {}
        elements_list = []
        for element, unique_point in zip(elements_map, unique_points):
            key = (np.around(unique_point[0], decimals=decimals), np.around(unique_point[1], decimals=decimals))
            if element in elements_dict:
                elements_dict[element][key] = unique_points_dict[key]
            else:
                elements_dict[element] = {key: unique_points_dict[key]}
                elements_list.append(element)
        logging.info('Only {}% of the elements need to be loaded.'.format(np.round(100*len(elements_list)/len(unique_points))))

        # Create a file map and a nag map for elements
        file_element_map = np.searchsorted(self.element_groups_info[group]['metadata']['elements_index_limits'], elements_list, side='right') - 1
        nag_map = []
        for element in elements_list:
            nag_map.append(self.element_groups_info[group]['metadata']['list_element_na'][element][2])
        logging.info('Created file and nag map')

        # Create file dict
        main_dict = {}
        file_list = []
        for file, element, nag in zip(file_element_map, elements_list, nag_map):
            if file in main_dict:
                if nag in main_dict[file]:
                    main_dict[file][nag][element] = elements_dict[element]
                else:
                    main_dict[file][nag] = {element: elements_dict[element]}
            else:
                main_dict[file] = {nag: {element: elements_dict[element]}}
                file_list.append(file)
        logging.info('Created main dictionary')

        # Go file by file, nag by nag
        logging.info('---START LOADING---')
        total_iterations = 0
        for file in main_dict.values():
            total_iterations += len(file.keys())
        with tqdm(total=total_iterations, desc="Loading and interpolating", unit="file") as pbar:
            for file in main_dict.keys():
                for nag in main_dict[file].keys():
                    # Grab the indices of the elements in the self.list_element_coords
                    elements = list(main_dict[file][nag].keys())
                    elements_in_file_nag = [self.element_groups_info[group]['metadata']['list_element_na'][element][3] for element in elements]

                    # Find problematic elements
                    problematic_elements = [] # contains the indices of the problematic elements in the local element list
                    proof = []
                    for index, element in enumerate(self.element_groups_info[group]['metadata']['list_element_coords'][elements]):
                        s = element[:,0]
                        z = element[:,1]
                        points = cart2polar(s,z)
                        r = points[[0,1,2],0]
                        theta = points[[0,3,6],1]
                        r_grid, theta_grid = np.meshgrid(r, theta)
                        expected_points = np.column_stack((r_grid.ravel(), theta_grid.ravel()))
                        if not np.allclose(points, expected_points, rtol=1e-6):
                            problematic_elements.append(index)
                            proof.append(points)
                    #self.plot_mesh(np.array(elements)[problematic_elements])
                    
                    # Read the data from the file
                    logging.info('Loading raw data.')
                    data = self.element_groups_info[group]['metadata']['files'][file]['data_wave__NaG=%d' % nag][elements_in_file_nag][:,:,:,channel_slices,time_slices].data

                    # Data expansion for unique inplane points
                    logging.info('Expanding data to all unique inplane points.')
                    in_element_repetitions = [len(inplane_points) for inplane_points in main_dict[file][nag].values()]
                    # may delete later
                    map_of_problematique = np.zeros(len(elements))
                    map_of_problematique[problematic_elements] = 1
                    if max(in_element_repetitions) == 1:
                        # This is the case where in each element there is only one
                        # inplane point where we need to run the inplane
                        # interpolation
                        expanded_data = data # no expansion occurs in this case
                        expanded_map_of_problematique = map_of_problematique
                    else:
                        # If there are multiple unique inplane points in some
                        # elements, we need to expand the data
                        final_shape = (np.sum(np.array(in_element_repetitions)),) + data.shape[1:]
                        expanded_data = np.empty(final_shape)
                        expanded_map_of_problematique = np.zeros(final_shape[0])
                        expanded_index = 0
                        for index, repetitions in enumerate(in_element_repetitions):
                            if repetitions == 1:
                                expanded_data[expanded_index] = data[index]
                                expanded_map_of_problematique[expanded_index] = map_of_problematique[index]
                                expanded_index += 1
                            else:
                                for _ in range(repetitions):
                                    expanded_data[expanded_index] = data[index]
                                    expanded_map_of_problematique[expanded_index] = map_of_problematique[index]
                                    expanded_index += 1
                    map_of_problematique = np.where(expanded_map_of_problematique == 1)[0]

                    # Find the inplane coords for each element
                    logging.info('Get unique inplane coords')
                    inplane_coords = []
                    for sub_dict in main_dict[file][nag].values():
                        for key in sub_dict.keys():
                            inplane_coords.append(key)

                    # Transform to polar coords
                    s, z = zip(*inplane_coords)
                    s = np.array(s)
                    z = np.array(z)
                    inplane_coords = cart2polar(s, z)

                    # Find the r and theta vectors for each element and construct
                    # lagrange interpolation matrix

                    # Expand coords of element GLL points to unique inplane points
                    logging.info('Expand element coords.')
                    points_of_interest = self.element_groups_info[group]['metadata']['list_element_coords'][elements][:,[0,1,2,3,6],:]
                    final_shape = (np.sum(np.array(in_element_repetitions)),) +  points_of_interest.shape[1:]
                    expanded_points_of_interest = np.empty(final_shape)
                    expanded_index = 0
                    for index, repetitions in enumerate(in_element_repetitions):
                        if repetitions == 1:
                            expanded_points_of_interest[expanded_index] = points_of_interest[index]
                            expanded_index += 1
                        else:
                            for _ in range(repetitions):
                                expanded_points_of_interest[expanded_index] = points_of_interest[index]
                                expanded_index += 1
                    points_of_interest = expanded_points_of_interest
                    # remove the points of interest located in problematique elements
                    points_of_interest = np.delete(points_of_interest, map_of_problematique, axis=0)
                    good_inplane_coords = np.delete(inplane_coords, map_of_problematique, axis=0)
                    original_shape = points_of_interest.shape
                    points_of_interest = points_of_interest.reshape((points_of_interest.shape[0]*points_of_interest.shape[1], 2))
                    points_of_interest = cart2polar(points_of_interest[:,0], points_of_interest[:,1]).reshape(original_shape)
                    GLL_rads = points_of_interest[:,[0,1,2],[0]]
                    GLL_thetas = points_of_interest[:,[2,3,4],[1]]
                    
                    # Compute interpolation weights (LIM)
                    logging.info('Compute LIM')
                    lr = np.array([
                        self._lagrange(good_inplane_coords[:,0], GLL_rads, i, lagrange_order) 
                        for i in range(lagrange_order)]).transpose()
                    ltheta = np.array([
                        self._lagrange(good_inplane_coords[:,1], GLL_thetas, i, lagrange_order) 
                        for i in range(lagrange_order)]).transpose()
                    LIM = np.array([
                        np.outer(ltheta_i, lr_i).flatten() 
                        for ltheta_i, lr_i in zip(ltheta, lr)])

                    # Manually add back into LIM some improvised weights at the
                    # locations within map_of_problematique
                    if len(LIM) == 0:
                        LIM = np.concatenate((LIM,np.array([0,0,0,0,1,0,0,0,0])), axis=0)
                        LIM = LIM.reshape((1,len(LIM)))
                        for location in map_of_problematique[1:]:
                            LIM = np.insert(LIM, location, np.array([0,0,0,0,1,0,0,0,0]), axis=0)
                    else:
                        for location in map_of_problematique:
                            LIM = np.insert(LIM, location, np.array([0,0,0,0,1,0,0,0,0]), axis=0)
                    
                    # Interpolate
                    logging.info('Inplane interpolate')
                    inplane_interpolated_data = np.sum(expanded_data * LIM[:,np.newaxis,:,np.newaxis,np.newaxis], axis=2)

                    # Add Fourier Coefficients
                    inplane_point_repetitions = []
                    for inplane_points in main_dict[file][nag].values():
                        for azimuthal_points in inplane_points.values():
                            inplane_point_repetitions.append(len(azimuthal_points))                        
                    
                    # expand to point level
                    logging.info('Expanding data to points')
                    if max(inplane_point_repetitions) == 1:
                        expanded_interpolated_data = inplane_interpolated_data
                    else:
                        final_shape = (np.sum(np.array(inplane_point_repetitions)),) + inplane_interpolated_data.shape[1:]
                        expanded_interpolated_data = np.empty(final_shape)
                        expanded_index = 0
                        for index, repetitions in enumerate(inplane_point_repetitions):
                            if repetitions == 1:
                                expanded_interpolated_data[expanded_index] = inplane_interpolated_data[index]
                                expanded_index += 1
                            else:
                                for _ in range(repetitions):
                                    expanded_interpolated_data[expanded_index] = inplane_interpolated_data[index]
                                    expanded_index += 1

                    # Fourier interpolation
                    logging.info('Fourier interpolation')
                    # Get the phis
                    phi = []
                    name_list = []
                    for sub_dict in main_dict[file][nag].values():
                        for sub_sub_dict in sub_dict.values():
                            for key in sub_sub_dict.keys():
                                name_list.append(key)
                                phi.append(sub_sub_dict[key][-1])
                    phi = np.array(phi)
                    # Set complex type
                    complex_type = expanded_interpolated_data.dtype if np.iscomplexobj(expanded_interpolated_data) else np.complex128
                    max_fourier_order = nag // 2
                    result = expanded_interpolated_data[:,0,:,:].copy()

                    for order in range(1, max_fourier_order + 1):
                        coeff = np.zeros(result.shape, dtype=complex_type)
                        # Real part
                        coeff.real = expanded_interpolated_data[:,order * 2 - 1,:,:]
                        # Complex part of Fourier coefficients
                        if order * 2 < nag:  # Check for Nyquist
                            coeff.imag += expanded_interpolated_data[:,order * 2,:,:]
                        result += (2.0 * np.exp(1j * order * phi)[:, np.newaxis, np.newaxis] * coeff).real

                    # Place the result in the final result ndarray
                    final_result[name_list,:,:] = result

                    pbar.update(1)
        return final_result
    
    
    def load_data_on_mesh(self, mesh:Mesh, channels: list, time_slices: list):
        """
        Load data on a mesh.
        Not using multi-processing!

        Args:
            source_location (list): The source location [depth, latitude, longitude] in degrees.
            station_location (list): The station location [depth, latitude, longitude] in degrees.
            R_max (float): The maximum radius for the slice in Earth radii.
            R_min (float): The minimum radius for the slice in Earth radii.
            resolution (int): The resolution of the slice (number of points along each dimension).
            channels (list): The channels of data to load.
            return_slice (bool, optional): Whether to return additional slice information. 
                                        Defaults to False.

        Returns:
            numpy.ndarray or list: An ndarray containing the loaded data on the slice,
                                and optionally, additional slice information.

        """                                                                    
        inplane_field = np.full((mesh.resolution, mesh.resolution, len(channels), len(time_slices)), np.NaN)
        data = self.load_data(points=mesh.points, frame='geographic', coords='spherical', in_deg=False,
                              channels=channels, time_slices=time_slices)
        index = 0
        for [index1, index2], _ in zip(mesh.indices, mesh.points):
            inplane_field[index1, index2, :, :] = data[index]
            index += 1

        return inplane_field


    def animation(self, source_location: np.ndarray=None, station_location: np.ndarray=None, channels: list=['U'],
                          name: str='video', video_duration: int=20, frame_rate: int=10,
                          resolution: int=100, domains: list=None,
                          lower_range: float=0.6, upper_range: float=0.9999,
                          paralel_processing: bool=False, mesh_type: str='slice'):
        """
        Generate an animation representing seismic data on a slice frame.

        Args:
            source_location (list): The coordinates [rad, lat, lon] of the seismic source in the Earth frame in radians.
            station_location (list): The coordinates [rad, lat, lon] of the station location in the Earth frame in radians.
            name (str, optional): The name of the output video file. Defaults to 'video'.
            video_duration (int, optional): The duration of the video in seconds. Defaults to 20.
            frame_rate (int, optional): The number of frames per second in the video. Defaults to 10.
            resolution (int, optional): The resolution of the slice mesh. Defaults to 100.
            R_min (float, optional): The minimum radius for data inclusion. Defaults to 0.
            R_max (float, optional): The maximum radius for data inclusion. Defaults to 6371000.
            lower_range (float, optional): The lower percentile range for the colorbar intensity. Defaults to 0.5.

        Returns:
            None
        """
        # Create default domains if None were given
        if domains is None:
            domains = []
            for element_group in self.element_groups_info.values():
                domains.append(element_group['elements']['vertical_range'] + [-2*np.pi, 2*np.pi])
        domains = np.array(domains)

        # Create source and station if none were given
        R_max = np.max(domains[:,1])
        R_min = np.min(domains[:,0])
        if source_location is None and station_location is None:
            source_location = np.array([R_max, 0, 0])
            station_location = np.array([R_max, 0, np.radians(30)])
        
        # Get time slices from frame rate and video_duration assuming that the
        # video will include the entire time axis. We also assume that each
        # element group that will be plotted has the same time axis
        # Check if all time axes are the same
        data_time_arrays = [self.element_groups_info[key]['metadata']['data_time'] 
                            for key in self.element_groups_info]
        all_data_time_same = all(np.array_equal(data_time_arrays[0], arr) 
                                 for arr in data_time_arrays[1:])
        if not all_data_time_same:
            logging.error('Not all element groups have the same time axis.')
            sys.exit()
        data_time = data_time_arrays[0]
        no_frames = frame_rate*video_duration
        time_slices = np.round(np.linspace(0, len(data_time) - 1, no_frames)).astype(int)

        # Load the data
        logging.info('Loading data')
        if mesh_type == 'slice':
            mesh = SliceMesh(source_location, station_location, domains, resolution)
        inplane_field = self.load_data_on_mesh(mesh, channels, time_slices)

        # Create animation
        logging.info('Create animation')
        # Create a figure and axis
        num_subplots = len(channels)
        num_rows = int(np.ceil(num_subplots / 3))
        num_cols = 3 if num_subplots > 1 else 1

        # Make a cbar
        cbar_min = []
        cbar_max = []
        for channel_slice in range(len(channels)):
            try:
                with np.errstate(divide='ignore', invalid='ignore'):
                    cbar_min_temp, cbar_max_temp = self._find_range(np.log10(np.abs(inplane_field[:,:,channel_slice,:])), lower_range, upper_range)
            except:
                cbar_min_temp, cbar_max_temp = [0, 0.1]
            cbar_min.append(cbar_min_temp)
            cbar_max.append(cbar_max_temp)

        # Set font size
        plt.rcParams.update({'font.size': 6})
        
        # Create a figure and axes
        fig, axes = plt.subplots(num_rows, num_cols, dpi=300)

        # Replace -inf values with small values for plotting purposes only Also
        # set to nan all values outside of the R_max circle and inside the R_min
        # circle
        processed_values = np.log10(np.abs(inplane_field))
        processed_values[np.isneginf(processed_values)] = min(cbar_min) - 1
        # Replace values outside the circle with NaN
        distance_from_center = np.sqrt(mesh.inplane_DIM1**2 + mesh.inplane_DIM2**2)


        # Find out which discontinuities (from base model) appear in the animation
        discontinuities_to_plot = [discontinuity for discontinuity in self.base_model['DISCONTINUITIES'] if R_min <= discontinuity <= R_max]
        for channel_slice, ax in enumerate(np.ravel(axes)):
            if channel_slice < len(channels):
                ax.set_aspect('equal')
                contour = ax.contourf(mesh.inplane_DIM1, mesh.inplane_DIM2, 
                                    processed_values[:, :, channel_slice, 0], 
                                    levels=np.linspace(cbar_min[channel_slice], cbar_max[channel_slice], 100), 
                                    cmap='RdBu_r', extend='both')
                ax.scatter(np.dot(mesh.point1, mesh.base1), np.dot(mesh.point1, mesh.base2))
                ax.scatter(np.dot(mesh.point2, mesh.base1), np.dot(mesh.point2, mesh.base2))
                ax.set_title(f'Subplot {channels[channel_slice]}')

                # Create a colorbar for each subplot
                cbar = plt.colorbar(contour, ax=ax)
                cbar_ticks = np.linspace(cbar_min[channel_slice], cbar_max[channel_slice], 5)
                cbar_ticklabels = ['{:0.1e}'.format(cbar_tick) for cbar_tick in cbar_ticks]
                cbar.set_ticks(cbar_ticks)
                cbar.set_ticklabels(cbar_ticklabels)
                cbar.set_label('Intensity')
            else:
                ax.axis('off')

        def update(frame):
            for channel_slice, ax in enumerate(np.ravel(axes)):
                if channel_slice < len(channels):
                    ax.cla()
                    ax.set_aspect('equal')
                    # Color everything outside the circle in white
                    outside_circle = plt.Circle((0, 0), R_max, color='white', fill=True)
                    ax.add_artist(outside_circle)
                    
                    contour = ax.contourf(mesh.inplane_DIM1, mesh.inplane_DIM2, 
                                        processed_values[:, :, channel_slice, frame], 
                                        levels=np.linspace(cbar_min[channel_slice], cbar_max[channel_slice], 100), 
                                        cmap='RdBu_r', extend='both')
                    ax.scatter(np.dot(mesh.point1, mesh.base1), np.dot(mesh.point1, mesh.base2))
                    ax.scatter(np.dot(mesh.point2, mesh.base1), np.dot(mesh.point2, mesh.base2))
                    ax.set_title(f'Subplot {channels[channel_slice]}')
                    
                    # Add a circle with radius R to the plot
                    for r in discontinuities_to_plot:
                        circle = plt.Circle((0, 0), r, color='black', fill=False)
                        ax.add_artist(circle)
                else:
                    ax.axis('off')
            print(100 * frame / no_frames, '%')
            return contour

        # Adjust spacing between subplots
        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.3, hspace=0.3)

        # Create the animation
        ani = animation.FuncAnimation(fig, update, frames=video_duration * frame_rate, interval=1e3 / frame_rate)
        ani.save(self.path_to_elements_output + '/' + name + '_animation.mp4', writer='ffmpeg')


    def _point_not_in_output_domain(self, point: list) -> bool:
        # The point must be given as [rad, lat, lon] in radians and in the
        # geographical frame and spherical coords

        # Transfrom point into spherical coords in the source frame
        rad, lat, _ = cart2sph(cart_geo2cart_src(sph2cart(point), rotation_matrix=self.rotation_matrix))
        # In the inparam.output the angle for the horizontal range is actually
        # the colatitude in the source frame therefore we must transform the
        # latitude to colatitude
        colat = np.pi/2 - lat
        if self.vertical_range[0] > rad or rad > self.vertical_range[1] \
            or colat < self.horizontal_range[0] or colat > self.horizontal_range[1]:
            return True
        else:
            return False


    def _read_element_metadata(self, element_group_name: str):
        """Reads a folder that contains the element output files from Axisem3D
        and outputs the metadata needed to access any data point from the mesh.

        Returns:
            na_grid (numpy array): A 1D array that contains all "Nr"s used in the 
                                Fourier expansions in the D domain.
            data_time (np array): Global time steps of the simulation.
            list_element_na (np array): For each element, it gives a 1D array that
                                        contains:
                                        1. Element tag in the mesh
                                        2. Actual "Nr"
                                        3. Stored "Nr" (in case you didn't want to store
                                        all the Nrs)
                                        4. Element index in the data (local)
                                        5. Element index in the data (global)
            list_element_coords (np array): For each element, for each grid point,
                                            gives the coordinates in the D domain as
                                            (s, z).
            dict_list_element (dict): Lists of element tags arranged by Nr in the dict.
            nc_files (list): List of opened netCDF files containing the element output data.
            elements_index_limits (list): List of element index limits for each netCDF file.
            detailed_channels (list): List of detailed channel names.

        Note:
            This method assumes that the element output files are stored in the
            `path_to_elements_output` directory.
        """      
        ################ open files ################
        # filenames (sorted correctly)
        path_to_element_group = os.path.join(self.path_to_elements_output, element_group_name)
        nc_fnames = sorted([f for f in os.listdir(path_to_element_group) if 'axisem3d_synthetics.nc' in f])
        
        # open files
        nc_files = []
        for nc_fname in nc_fnames:
            nc_files.append(xr.open_dataset(os.path.join(path_to_element_group, nc_fname)))
        
        ################ variables that are the same in the datasets ################
        # read Na grid (all azimuthal dimensions)
        na_grid = nc_files[0].data_vars['list_na_grid'].values.astype(int)

        # read time
        data_time = nc_files[0].data_vars['data_time'].values
        
        ################ variables to be concatenated over the datasets minud the data itself################
        # define empty lists of xarray.DataArray objects
        xda_list_element_na = []
        xda_list_element_coords = []
        dict_xda_list_element = {}
        detailed_channels = [str_byte.decode('utf-8') for str_byte in nc_files[0].list_channel.data]
        elements_index_limits = [0]
        index_limit = 0
        ######dict_xda_data_wave = {}
        for nag in na_grid:
            dict_xda_list_element[nag] = []
        
        # loop over nc files
        for i, nc_file in enumerate(nc_files):
            # append DataArrays
            index_limit += nc_file.sizes['dim_element']
            elements_index_limits.append(index_limit)
            xda_list_element_na.append(nc_file.data_vars['list_element_na'])
            xda_list_element_coords.append(nc_file.data_vars['list_element_coords'])
            for nag in na_grid:
                dict_xda_list_element[nag].append(nc_file.data_vars['list_element__NaG=%d' % nag])

        # concat xarray.DataArray
        xda_list_element_na = xr.concat(xda_list_element_na, dim='dim_element')
        xda_list_element_coords = xr.concat(xda_list_element_coords, dim='dim_element')
        for nag in na_grid:
            dict_xda_list_element[nag] = xr.concat(dict_xda_list_element[nag], dim='dim_element__NaG=%d' % nag)

        # read data to numpy.ndarray
        list_element_na = xda_list_element_na.values.astype(int)
        list_element_coords = xda_list_element_coords.values
        dict_list_element = {}
        for nag in na_grid:
            dict_list_element[nag] = dict_xda_list_element[nag].values.astype(int)

        ############### return ################
        # Here we return the files only because in this format they are not being loaded into RAM
        # Since these files are huge we prefer to load into ram only the file where the data that we 
        # want is located and then close the file. 
        return na_grid, data_time, list_element_na, \
               list_element_coords, dict_list_element, \
               nc_files, elements_index_limits, \
               detailed_channels


    def _compute_rotation_matrix(self):
        """Computes the rotation matrix that aligns the z axis with the source axis

        Returns:
            np.ndarray: 3D rotation matrix
        """
        # get real earth coordinates of the sources
        colatitude = np.pi/2 - np.deg2rad(self.source_lat)
        longitude = np.deg2rad(self.source_lon)

        # rotation matrix into the source frame (based on Tarje's PhD)
        return np.asarray([[np.cos(colatitude) * np.cos(longitude), -np.sin(longitude), np.sin(colatitude) * np.cos(longitude)],
                           [np.cos(colatitude) * np.sin(longitude), np.cos(longitude), np.sin(colatitude) * np.sin(longitude)],
                           [-np.sin(colatitude), 0, np.cos(colatitude)]])


    def _lagrange(self, evaluation_points, GLL_points, i, order):
        """ Lagrange function implementation
        """
        value = 1
        for j in range(order):
            if i != j:
                try:
                    with warnings.catch_warnings(record=True) as w:
                        value *= (evaluation_points - GLL_points[:,j]) / (GLL_points[:,i] - GLL_points[:,j])
                    if w:
                        # A warning occurred, print it along with the traceback
                        warning = w[-1]
                        print(f"Warning: {warning.message}")
                except Exception as e:
                    # In case of any exception, print the error message
                    print(f"Error: {str(e)}")
        return value
            

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


    def _channel_slices(self, channels, group):
        # Get channel slices from channels
        detailed_channels = self.element_groups_info[group]['metadata']['detailed_channels']
        if isinstance(channels, list) and all(ch in detailed_channels for ch in channels):
            return [detailed_channels.index(ch) for ch in channels]

        if channels is not None:
            if(self._check_elements(channels, self.element_groups_info[group]['wavefields']['channels'])):
                # Filter by channel chosen
                channel_slices = []
                for channel in channels:
                    channel_slices += [index for index, element in enumerate(detailed_channels) if element.startswith(channel)]
            else:
                raise Exception('Only the following channels exist: ' + ', '.join(self.element_groups_info[group]['wavefields']['channels']))
        else:
            channel_slices = None

        return channel_slices


    def _find_simulation_path(self, path: str):
        """

        Args:
            path_to_station_file (str): path to station.txt file

        Returns:
            parent_directory
        """
        current_directory = os.path.abspath(path)
        while True:
            parent_directory = os.path.dirname(current_directory)
            if 'output' in os.path.basename(current_directory):
                return parent_directory
            elif current_directory == parent_directory:
                # Reached the root directory, "output" directory not found
                return None
            current_directory = parent_directory


    def _check_elements(self, list1, list2):
        """Checks if all elements in list1 can be found in list2.

        Args:
            list1 (list): The first list.
            list2 (list): The second list.

        Returns:
            bool: True if all elements in list1 are found in list2, False otherwise.
            list: List of elements from list1 that are not found in list2.
        """
        missing_elements = [element for element in list1 if element not in list2]
        if len(missing_elements) == 0:
            return True
        else:
            return False