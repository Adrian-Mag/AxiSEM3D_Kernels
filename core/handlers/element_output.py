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

from .axisem3d_output import AxiSEM3DOutput
from ...aux.coordinate_transforms import sph2cart, cart2sph, cart2polar, cart_geo2cart_src, cart2cyl

#warnings.filterwarnings("error")

class ElementOutput(AxiSEM3DOutput):
    def __init__(self, path_to_element_output:str) -> None:
        """Initializes the ElementOutput object for the given path to the element output directory.

        Args:
            path_to_element_output (str): Path to the element output directory.
            element_group (str, optional): Name of the element group. If None, the first element group found will be used.

        Attributes:
            path_to_elements_output (str): Path to the element output directory.
            na_grid (numpy.ndarray): NA grid information.
            data_time (numpy.ndarray): Data time information.
            list_element_na (numpy.ndarray): List of element NA values.
            list_element_coords (numpy.ndarray): List of element coordinates.
            dict_list_element (dict): Dictionary of list of elements.
            files (list): List of element output files.
            elements_index_limits (list): List containing element index limits.
            rotation_matrix (numpy.ndarray): Rotation matrix.
            coordinate_frame (str): Coordinate frame of the wavefields.
            channels (list): List of wavefield channels.
            detailed_channels (list): List of channels component by component
            GLL_points_one_edge (str): Grid format for the in-plane coordinates.
            source_lat (float): Latitude of the event located on the axis.
            source_lon (float): Longitude of the event located on the axis.
        """
        path_to_simulation = self._find_simulation_path(path_to_element_output)
        super().__init__(path_to_simulation)

        # If the output file has multiple element groups, you need to know which
        # one you are using, therefore we store its name
        self.element_group_name = os.path.basename(path_to_element_output)

        # Get all the data from the output model
        with open(self.inparam_output, 'r') as file:
            output_yaml = yaml.load(file, Loader=yaml.FullLoader)
            for dictionary in output_yaml['list_of_element_groups']:
                if self.element_group_name in dictionary:
                    element_group = dictionary.get(self.element_group_name, {})
                    break
            self.horizontal_range = element_group.get('elements', {}).get('horizontal_range')
            self.vertical_range = list(map(float, element_group.get('elements', {}).get('vertical_range', [])))
            self.edge_dimension = element_group.get('inplane', {}).get('edge_dimension')
            self.edge_position = element_group.get('inplane', {}).get('edge_position')
            self.GLL_points_one_edge = element_group.get('inplane', {}).get('GLL_points_one_edge')
            self.phi_list = element_group.get('azimuthal', {}).get('phi_list')
            self.lat_lon_list = element_group.get('azimuthal', {}).get('lat_lon_list')
            self.na_space = element_group.get('azimuthal', {}).get('na_space')
            self.coordinate_frame = element_group.get('wavefields', {}).get('coordinate_frame')
            self.medium = element_group.get('wavefields', {}).get('medium')
            self.channels = element_group.get('wavefields', {}).get('channels')
            self.sampling_period = element_group.get('temporal', {}).get('sampling_period')
            self.time_window = element_group.get('temporal', {}).get('time_window')

        # Get lat lon of the event located on the axis
        with open(self.inparam_source, 'r') as file:
            source_yaml = yaml.load(file, Loader=yaml.FullLoader)
            source_name = list(source_yaml.get('list_of_sources', [{}])[0].keys())[0]
            # Assume a single point source
            source = source_yaml.get('list_of_sources', [{}])[0].get(source_name, {})
            self.source_lat, self.source_lon = source.get('location', {}).get('latitude_longitude', [])
            self.source_depth = float(source.get('location', {}).get('depth'))

        self.path_to_elements_output = path_to_element_output

        # Get metadata
        (
            self.na_grid,
            self.data_time,
            self.list_element_na,
            self.list_element_coords,
            self.dict_list_element,
            self.files,
            self.elements_index_limits,
            self.detailed_channels,
        ) = self._read_element_metadata()

        # Replace the numerical indicators of coordinates with letters based on the coordinate system
        self.detailed_channels = [
            element.replace('1', self.coordinate_frame[0]).replace('2', self.coordinate_frame[1]).replace('3', self.coordinate_frame[2])
            for element in self.detailed_channels
        ]

        self.rotation_matrix = self._compute_rotation_matrix()


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
        stream.write(obspyfy_path + '/' + self.element_group_name + '.mseed', format="MSEED") 


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
            for channel in self.detailed_channels:
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
            channels_list.append(self.detailed_channels)

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
            # get the data at this station (assuming RTZ components)
            wave_data = self.load_data_at_point(point=np.array([starad, stalat, stalon]),
                                                coord_in_deg=True,
                                                channels=channels, 
                                                time_slices=time_slices)
            # COnstruct metadata
            delta = self.data_time[1] - self.data_time[0]
            npts = len(self.data_time)
            network = station['network']
            station_name = station['name']
            if channels is not None:
                selected_detailed_channels = [element for element in self.detailed_channels \
                                            if any(element.startswith(prefix) for prefix in channels)]
            else:
                selected_detailed_channels = self.detailed_channels
            for chn_index, chn in enumerate(selected_detailed_channels):
                # form the traces at the channel level
                trace = obspy.Trace(wave_data[chn_index])
                trace.stats.delta = delta
                trace.stats.ntps = npts
                trace.stats.network = network
                trace.stats.station = station_name
                trace.stats.location = ''
                trace.stats.channel = chn
                trace.stats.starttime = obspy.UTCDateTime("1970-01-01T00:00:00.0Z") + self.data_time[0]
                stream.append(trace)

        return stream


    def stream(self, point: list, coord_in_deg: bool = True, 
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
        # Get time slices from time limits
        if time_limits is not None:
            time_slices = np.where((self.data_time >= time_limits[0]) & (self.data_time <= time_limits[1]))
        else:
            time_slices = None

        if coord_in_deg:
            point[1] = np.deg2rad(point[1])
            point[2] = np.deg2rad(point[2])

        # initiate stream that will hold data 
        stream = obspy.Stream()
        # get the data at this station (assuming RTZ components)
        wave_data = self.load_data_at_point(point=point,
                                            channels=channels, 
                                            time_slices=time_slices, 
                                            coord_in_deg=False)
        # Construct metadata 
        delta = self.data_time[1] - self.data_time[0]
        npts = len(self.data_time)
        network = str(np.random.randint(0, 100))
        station_name = str(np.random.randint(0, 100))

        if channels is not None:
                selected_detailed_channels = [element for element in self.detailed_channels \
                                            if any(element.startswith(prefix) for prefix in channels)]
        else:
            selected_detailed_channels = self.detailed_channels
        for chn_index, chn in enumerate(selected_detailed_channels):
            # form the traces at the channel level
            trace = obspy.Trace(wave_data[chn_index])
            trace.stats.delta = delta
            trace.stats.ntps = npts
            trace.stats.network = network
            trace.stats.station = station_name
            trace.stats.location = ''
            trace.stats.channel = chn
            trace.stats.starttime = obspy.UTCDateTime("1970-01-01T00:00:00.0Z") + self.data_time[0]
            stream.append(trace)

        return stream


    def load_data(self, points: np.ndarray, frame: str='geographic', 
                  coords: str='spherical', in_deg: bool=True,
                  channels: list=None, time_slices: list=None):
        # only options are geographic+spherical, geographic+cartesian,
        # source+cylindrical If only one point, we will make it an array of
        # array
        if len(points) == 3:
            points = points.reshape((1,3))
        
        # Get lagrange order
        if self.GLL_points_one_edge == [0,2,4]:
            lagrange_order = 3
            logging.info('Using lagrange order 3.')
        else:
            logging.error("No implementation for this output type.")
            raise NotImplementedError("No implementation for this output type.")

        # Make sure the data type of points is floats
        points = points.astype(np.float64)
        
        # Create channel and time slices
        if channels is None:
            channels = self.detailed_channels
        if time_slices is None:
            time_slices = np.arange(len(self.data_time))
        channel_slices = self._channel_slices(channels)
        final_result = np.ones((len(points),len(channels),len(time_slices)))
        
        # Transforms points to cylindrical coords in source frame
        if frame == 'geographic':
            if coords == 'spherical':
                if in_deg is True:
                    points[:, 1:] = np.deg2rad(points[:, 1:])
                points = sph2cart(points)
            points = cart2cyl(cart_geo2cart_src(points=points, 
                                                rotation_matrix=self.rotation_matrix))
            logging.info("Transformed points to cylindrical coordinates in source frame.")

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
        element_centers = self.list_element_coords[:, 4, :]  # Assuming the center point is at index 4
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
        file_element_map = np.searchsorted(self.elements_index_limits, elements_list, side='right') - 1
        nag_map = []
        for element in elements_list:
            nag_map.append(self.list_element_na[element][2])
        nag_list = list(set(nag_map))
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
                    elements_in_file_nag = [self.list_element_na[element][3] for element in elements]

                    # Find problematic elements
                    problematic_elements = [] # contains the indices of the problematic elements in the local element list
                    proof = []
                    for index, element in enumerate(self.list_element_coords[elements]):
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
                    self.plot_mesh(np.array(elements)[problematic_elements])
                    
                    # Read the data from the file
                    logging.info('Loading raw data.')
                    data = self.files[file]['data_wave__NaG=%d' % nag][elements_in_file_nag][:,:,:,channel_slices,time_slices].data

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
                    points_of_interest = self.list_element_coords[elements][:,[0,1,2,3,6],:]
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
    
    
    def load_data_on_slice(self, source_location: np.ndarray, station_location: np.ndarray, 
                            R_max: float, R_min: float, theta_min: float, theta_max: float, 
                            resolution: int, channels: list, time_slices: list, return_slice: bool=False):
        """
        Load data on a slice of points within a specified radius range and resolution.
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
        if return_slice is False:
            filtered_indices, filtered_slice_points = self._create_slice(source_location=source_location, 
                                                                         station_location=station_location, 
                                                                         R_min=R_min, R_max=R_max, 
                                                                         theta_min=theta_min, 
                                                                         theta_max=theta_max,
                                                                         resolution=resolution,
                                                                         return_slice=return_slice)
        else:
            filtered_indices, filtered_slice_points, \
            point1, point2, base1, base2, \
            inplane_DIM1, inplane_DIM2 = self._create_slice(source_location=source_location, 
                                                            station_location=station_location, 
                                                            R_min=R_min, R_max=R_max, 
                                                            theta_min=theta_min, 
                                                            theta_max=theta_max,
                                                            resolution=resolution,
                                                            return_slice=return_slice)
        inplane_field = np.zeros((resolution, resolution, len(channels), len(time_slices)))
        data = self.load_data(points=filtered_slice_points, frame='geographic', coords='spherical', in_deg=False,
                              channels=channels, time_slices=time_slices)
        index = 0
        for [index1, index2], point in zip(filtered_indices, filtered_slice_points):
            inplane_field[index1, index2, :, :] = data[index]
            index += 1

        if return_slice is False:
            return inplane_field
        else:
            return [inplane_field, point1, point2, 
                    base1, base2, 
                    inplane_DIM1, inplane_DIM2]


    def load_data_at_point(self, point: np.ndarray, coord_in_deg: bool = False, 
                        channels: list = None,
                        time_slices: list = None) -> np.ndarray:
        """
        Expand an in-plane point into the longitudinal direction using the Fourier expansion.

        Args:
            point (list): A list representing the point in geographical frame.
                        It should contain the following elements:
                        - radial position in meters (float)
                        - latitude in degrees/radians (float)
                        - longitude in degrees/radians (float)
            coord_in_deg (bool, optional): Specify whether the coordinates are in degrees.
                                        Defaults to True.
            channels (list, optional): List of channels to include.
                                    Defaults to None, which includes all channels.
            time_slices (list, optional): Points on the time axis where data should be loaded
                                        Defaults to None, which includes all times.
            fourier_order (int, optional): Maximum Fourier order.
                                        Defaults to None.

        Returns:
            np.ndarray: The result of the Fourier expansion, represented as a NumPy array.
        """

        # Make sure the degrees are turned to radians
        if coord_in_deg:
            point[1] = np.deg2rad(point[1])
            point[2] = np.deg2rad(point[2])

        # Check if the point provided is in the output domain
        if self._point_not_in_output_domain(point):
            print('A point that is not in the output domain has been found: ', point)

        # Transform the geographical frame spherical coord into the source frame cylindrical coords
        _, _, phi = cart2cyl(cart_geo2cart_src(sph2cart(point), rotation_matrix=self.rotation_matrix))

        # Get channel slices from channels
        channel_slices = self._channel_slices(channels=channels)

        # Interpolate the data in-plane
        interpolated_data = self._inplane_interpolation(point=point, channel_slices=channel_slices, 
                                                        time_slices=time_slices)

        # Set complex type
        complex_type = interpolated_data.dtype if np.iscomplexobj(interpolated_data) else np.complex128

        # Find max Fourier order
        max_fourier_order = len(interpolated_data[:, 0, 0]) // 2

        # Initialize result with 0th order
        result = interpolated_data[0].copy()

        # Add higher orders
        for order in range(1, max_fourier_order + 1):
            coeff = np.zeros(result.shape, dtype=complex_type)
            # Real part
            coeff.real = interpolated_data[order * 2 - 1]
            # Complex part of Fourier coefficients
            if order * 2 < len(interpolated_data):  # Check for Nyquist
                coeff.imag += interpolated_data[order * 2]
            result += (2.0 * np.exp(1j * order * phi) * coeff).real

        if np.max(np.abs(result)) > 1:
            print(point) 

        return result


    def animation(self, source_location: np.ndarray=None, station_location: np.ndarray=None, channels: list=['U'],
                          name: str='video', video_duration: int=20, frame_rate: int=10,
                          resolution: int=100, R_min: float=None, R_max: float=None,
                          theta_min: float=-np.pi, theta_max: float=np.pi,
                          lower_range: float=0.6, upper_range: float=0.9999,
                          paralel_processing: bool=True, batch_size: int=1000):
        """
        Generate an animation representing seismic data on a slice frame.

        Args:
            source_location (list): The coordinates [depth, lat, lon] of the seismic source in the Earth frame.
            station_location (list): The coordinates [depth, lat, lon] of the station location in the Earth frame.
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
        # Fill in the args with default to None
        if R_min is None:
            R_min = self.vertical_range[0]
        if R_max is None:
            R_max = self.vertical_range[1]

        # Auto points
        if source_location is None and station_location is None:
            source_location = np.array([self.Earth_Radius - R_max, 0, 0])
            station_location = np.array([self.Earth_Radius - R_max, 0, 30])
        
        # Get time slices from frame rate and video_duration assuming that the
        # video will include the entire time axis 
        no_frames = frame_rate*video_duration
        time_slices = np.round(np.linspace(0, len(self.data_time) - 1, no_frames)).astype(int)

        logging.info('Loading data')
        if paralel_processing is True:
            pass
        else:
            start_time = time.time()
            inplane_field, point1, point2, \
            base1, base2, inplane_DIM1, \
            inplane_DIM2 = self.load_data_on_slice(source_location=source_location, 
                                                   station_location=station_location, 
                                                   R_max=R_max, R_min=R_min, 
                                                   theta_min=theta_min, theta_max=theta_max,
                                                   resolution=resolution, channels=channels, 
                                                   time_slices=time_slices, return_slice=True)
            end_time = time.time()

        logging.info('Create animation')
        # Create a figure and axis
        num_subplots = len(channels)
        num_rows = int(np.ceil(num_subplots / 2))
        num_cols = 2 if num_subplots > 1 else 1

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
        distance_from_center = np.sqrt(inplane_DIM1**2 + inplane_DIM2**2)
        processed_values[distance_from_center > R_max] = np.nan
        processed_values[distance_from_center < R_min] = np.nan

        # Find out which discontinuities (from base model) appear in the animation
        discontinuities_to_plot = [discontinuity for discontinuity in self.base_model['DISCONTINUITIES'] if R_min <= discontinuity <= R_max]
        for channel_slice, ax in enumerate(np.ravel(axes)):
            if channel_slice < len(channels):
                ax.set_aspect('equal')
                contour = ax.contourf(inplane_DIM1, inplane_DIM2, 
                                    processed_values[:, :, channel_slice, 0], 
                                    levels=np.linspace(cbar_min[channel_slice], cbar_max[channel_slice], 100), 
                                    cmap='RdBu_r', extend='both')
                ax.scatter(np.dot(point1, base1), np.dot(point1, base2))
                ax.scatter(np.dot(point2, base1), np.dot(point2, base2))
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
                    
                    contour = ax.contourf(inplane_DIM1, inplane_DIM2, 
                                        processed_values[:, :, channel_slice, frame], 
                                        levels=np.linspace(cbar_min[channel_slice], cbar_max[channel_slice], 100), 
                                        cmap='RdBu_r', extend='both')
                    ax.scatter(np.dot(point1, base1), np.dot(point1, base2))
                    ax.scatter(np.dot(point2, base1), np.dot(point2, base2))
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
 

    def _inplane_interpolation(self, point: list, channel_slices: list = None, 
                              time_slices: list = None)-> np.ndarray:
        """Takes in a point in spherical coordinates in the real earth frame
        and outputs the displacement data in time for all the available channels
        in the form of a NumPy array.

        Args:
            point (list): A list representing the point in spherical coordinates in geographical frame.
                        It should contain the following elements:
                        - radial position in meters (float)
                        - latitude in radians (float)
                        - longitude in radians (float)
            channels (list, optional): List of channels to include. Defaults to None, which includes all channels.
            time_limits (list, optional): Time limits for the data. It should be a list with two elements:
                                        - start time in seconds (float)
                                        - end time in seconds (float)
                                        Defaults to None, which includes all times.

        Returns:
            np.ndarray: The interpolated displacement data in time for all available channels,
                        represented as a NumPy array.
        """      

        # Transform geographical to cylindrical coords in source frame
        s, z, _ = cart2cyl(cart_geo2cart_src(sph2cart(point), rotation_matrix=self.rotation_matrix))
        # spherical coordinates will be used for the GLL interpolation
        [r, theta] = cart2polar(s,z)[0]

        if self.GLL_points_one_edge == [0,2,4]:
            # The of the element are positioned like this (GLL point)
            # The numbers inbetween the points are the sub-element numbers
            # ^r
            # | (2)-(5)-(8)
            # |  | 1 | 3 |
            # | (1)-(4)-(7)
            # |  | 0 | 2 |
            # | (0)-(3)-(6)
            #  ____________>theta

            # find the difference vector between our chosen point
            # and the center GLL point of every element
            difference_vectors = self.list_element_coords[:,4,0:2] - [s, z]
            
            # find the index of the central GLL point that is closest to our point 
            element_index = np.argmin((difference_vectors*difference_vectors).sum(axis=1))
            
            # grab the information about the element whose center we just found
            element_na = self.list_element_na[element_index]
            for i in range(len(self.elements_index_limits) - 1):
                if self.elements_index_limits[i] <= element_index < self.elements_index_limits[i+1]:
                    file_index = i
                    break
            # get the element points
            element_points = self.list_element_coords[element_index]
            radial_element_GLL_points = cart2polar(element_points[[0,1,2]][:,0], 
                                                    element_points[[0,1,2]][:,1])[:,0]
            theta_element_GLL_points = cart2polar(element_points[[0,3,6]][:,0], 
                                                    element_points[[0,3,6]][:,1])[:,1]

            # Now we get the data
            data_wave = self._read_element_data(element_na=element_na, file_index=file_index,
                                                channel_slices=channel_slices, time_slices=time_slices)
            # finally we interpolate at our point
            interpolated_data = np.zeros(data_wave[:,0,:,:].shape)
            # Now we interpolate using GLLelement_na, file_index, channels, time_limits
            for i in range(3):
                for j in range(3):
                    interpolated_data += self._lagrange(r, np.array([radial_element_GLL_points]), j, 3) * \
                    self._lagrange(theta, np.array([theta_element_GLL_points]), j, 3) * data_wave[:,3*i+j,:,:]

        return interpolated_data


    def _read_element_metadata(self):
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
        nc_fnames = sorted([f for f in os.listdir(self.path_to_elements_output) if 'axisem3d_synthetics.nc' in f])
        
        # open files
        nc_files = []
        for nc_fname in nc_fnames:
            nc_files.append(xr.open_dataset(self.path_to_elements_output + '/' + nc_fname))
        
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
        updated_array = []
        for element in detailed_channels:
            letter = element[0]
            digits = ''.join(sorted(element[1:]))
            updated_array.append(letter + digits)
        detailed_channels = updated_array
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


    def _read_element_data(self, element_na, file_index: int, 
                           channel_slices: list=None, time_slices: list=None):
        """Reads the element data from the specified file and returns the wave data.

        Args:
            element_na (tuple): Element information containing:
                - Element tag in the mesh
                - Actual "Nr"
                - Stored "Nr" (in case you didn't want to store all the Nrs)
                - Element index in the data (local)
                - Element index in the data (global)
            file_index (int): Index of the file containing the element data.
            channels (list, optional): List of channels to include. Defaults to None.
            time_limits (list, optional): List of time limits [t_min, t_max]. Defaults to None.

        Returns:
            np.ndarray: The wave data.

        Raises:
            Exception: If the specified channels or time limits are not available.

        Note:
            - If `channels` and `time_limits` are both None, the entire wave data is returned.
            - If only `time_limits` is provided, the wave data is filtered by the specified time range.
            - If only `channels` are provided, the wave data is filtered by the specified channels.
            - If both `channels` and `time_limits` are provided, the wave data is filtered by both.

        Note that the wave data is assumed to be stored in the `files` attribute, which is a list of opened netCDF files.
        """
        if channel_slices is None and time_slices is None:
            wave_data = self.files[file_index]['data_wave__NaG=%d' % element_na[2]][element_na[3]].values
        elif channel_slices is not None and time_slices is None:
            wave_data = self.files[file_index]['data_wave__NaG=%d' % element_na[2]][element_na[3]][:, :, channel_slices, :].values
        elif channel_slices is None and time_slices is not None:
            wave_data = self.files[file_index]['data_wave__NaG=%d' % element_na[2]][element_na[3]][:, :, :, time_slices].values
        elif channel_slices is not None and time_slices is not None:
            wave_data = self.files[file_index]['data_wave__NaG=%d' % element_na[2]][element_na[3]][:, :, channel_slices, time_slices].values
        
        return wave_data


    def _create_slice(self, source_location: np.ndarray, station_location: np.ndarray, 
                      R_min: float, R_max: float, theta_min: float, theta_max: float,
                      resolution: int, return_slice: bool=False):
        """
        Create a mesh for a slice of Earth within a specified radius range and
        resolution.

        Args:
            source_location (np.ndarray): The source location [depth, latitude, longitude] in degrees.
            station_location (np.ndarray): The station location [depth, latitude, longitude] in degrees.
            R_max (float): The maximum radius for the slice in Earth radii.
            R_min (float): The minimum radius for the slice in Earth radii.
            resolution (int): The resolution of the slice (number of points along each dimension).
            return_slice (bool, optional): Whether to return additional slice information. 
                                        Defaults to False.

        Returns:
            list: A list containing filtered indices and slice points, and optionally, 
            additional slice information.

        """
        # Transform from depth lat lon in deg to rad lat lon in rad
        source_location[0] = self.Earth_Radius - source_location[0]
        source_location[1] = np.deg2rad(source_location[1])
        source_location[2] = np.deg2rad(source_location[2])
        station_location[0] = self.Earth_Radius - station_location[0]
        station_location[1] = np.deg2rad(station_location[1])
        station_location[2] = np.deg2rad(station_location[2])

        # Form vectors for the two points (Earth frame)
        point1 = sph2cart(source_location)
        point2 = sph2cart(station_location)

        # Do Gram-Schmidt orthogonalization to form slice basis (Earth frame)
        base1 = point1 / np.linalg.norm(point1)
        base2 = point2 - np.dot(point2, base1) * base1
        base2 /= np.linalg.norm(base2)

        # Generate index mesh
        indices_dim1 = np.arange(resolution)
        indices_dim2 = np.arange(resolution)

        # Generate in-plane mesh
        inplane_dim1 = np.linspace(-R_max, R_max, resolution)
        inplane_dim2 = np.linspace(-R_max, R_max, resolution)
        inplane_DIM1, inplane_DIM2 = np.meshgrid(inplane_dim1, inplane_dim2, indexing='ij')
        radii = np.sqrt(inplane_DIM1*inplane_DIM1 + inplane_DIM2*inplane_DIM2)
        thetas = np.arctan2(inplane_DIM2, inplane_DIM1)

        # Generate slice mesh points
        filtered_indices = []
        filtered_slice_points = []
        for index1 in indices_dim1:
            for index2 in indices_dim2:
                if radii[index1, index2] < R_max and radii[index1, index2] > R_min \
                    and thetas[index1, index2] < theta_max and thetas[index1, index2] > theta_min:
                    point = inplane_dim1[index1] * base1 + inplane_dim2[index2] * base2  # Slice frame -> Earth frame
                    filtered_slice_points.append(cart2sph(point))
                    filtered_indices.append([index1, index2])
        if return_slice is False:
            return [np.array(filtered_indices), np.array(filtered_slice_points)]
        else:
            return [np.array(filtered_indices), np.array(filtered_slice_points),
                    point1, point2, base1, base2, 
                    inplane_DIM1, inplane_DIM2]


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


    def _channel_slices(self, channels):
                # Get channel slices from channels
        if isinstance(channels, list) and all(ch in self.detailed_channels for ch in channels):
            return [self.detailed_channels.index(ch) for ch in channels]
    
        if channels is not None:
            if(self._check_elements(channels, self.channels)):
                # Filter by channel chosen
                channel_slices = []
                for channel in channels:
                    channel_slices += [index for index, element in enumerate(self.detailed_channels) if element.startswith(channel)]
            else:
                raise Exception('Only the following channels exist: ' + ', '.join(self.channels))
        else:
            channel_slices = None

        return channel_slices


    def _find_simulation_path(self, path: str):
        """Takes in the path to a station file used for axisem3d
        and returns a stream with the wavefields computed at all stations

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