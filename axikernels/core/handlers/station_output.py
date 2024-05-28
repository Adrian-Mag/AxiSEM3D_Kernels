"""
Class for handling axisem station output in netcdf format.
"""
import numpy as np
import obspy
from obspy import UTCDateTime
import pandas as pd
import netCDF4 as nc
import os
import yaml
from obspy.core.inventory import Inventory, Network, Station, Channel

from .axisem3d_output import AxiSEM3DOutput


class StationOutput(AxiSEM3DOutput):
    def __init__(self, path_to_station_output: str):
        """Class for handling axisem station output in netcdf format.

        Args:
            path (str): Path to station output directory of simultaion
        """
        path_to_simulation = self._find_simulation_path(path_to_station_output)
        super().__init__(path_to_simulation)

        # Name and path of station output
        self.path_to_station_output = path_to_station_output
        self.station_group_name = os.path.basename(path_to_station_output)
        # Load output files
        self._files = self._load_files()
        # Load netcdf data as Datasets
        self._nc_data = self._load_netcdf_datasets()
        # read the rank list file
        self._rank_list = self._load_rank_list()
        # Load time data
        self.data_time = self._get_time()
        # Load metadata
        self._load_metadata()

    def _load_metadata(self):
        with open(self.inparam_output, 'r') as file:
            output_yaml = yaml.load(file, Loader=yaml.FullLoader)
            station_group = output_yaml['list_of_station_groups'][0][self.station_group_name] # noqa
            self.coordinate_frame = station_group['wavefields']['coordinate_frame'] # noqa
            self.channels = station_group['wavefields']['channels']
            self._stations_file_name = station_group['locations']['station_file'] # noqa
            self._stations_file_path = self.path_to_simulation + '/input/' + self._stations_file_name # noqa
        self.detailed_channels = self._get_detailed_channels()
        self.detailed_channels = [element.replace('1', self.coordinate_frame[0]). # noqa
                                  replace('2', self.coordinate_frame[1]).
                                  replace('3', self.coordinate_frame[2])
                                  for element in self.detailed_channels]

    def _load_files(self):
        pattern = 'axisem3d_synthetics.nc.rank'
        files = [
            f for f in os.listdir(self.path_to_station_output) if pattern in f
            ]
        return files

    def _load_netcdf_datasets(self):
        netcdf_data = {
            filename: nc.Dataset(self.path_to_station_output + '/' + filename)
            for filename in self._files
            }
        return netcdf_data

    def _load_rank_list(self):
        rank_list = pd.read_csv(
            os.path.join(self.path_to_station_output, 'rank_station.info'),
            sep=' ',
            header=0,
            names=['MPI_RANK', 'STATION_KEY', 'STATION_INDEX_IN_RANK'],
        )
        return rank_list

    def load_data_at_station(self, network: str,
                             station_name: str,
                             channels: list = None,
                             time_limits: list = None):
        """Load wave data at a specific station.

        Args:
            network (str): Network name
            station_name (str): Station name
            channels (list, optional): List of channel names. Defaults to None.
            time_limits (list, optional): List of time limits [t_min, t_max].
            Defaults to None.

        Returns:
            np.ndarray: Wave data at the station
        """

        # Form the station key
        station_key = network + '.' + station_name
        # Find the file and location within the file of the station
        row = self._rank_list[self._rank_list['STATION_KEY'] == station_key]
        if row.empty:
            raise ValueError('The station ' + station_key + ' does not exist.')
        rank_index = row.values.tolist()[0][0]
        in_file_index = row.values.tolist()[0][2]
        filename = 'axisem3d_synthetics.nc.rank' + str(rank_index)
        # get wave data for this rank
        wave_data = nc.Dataset(
            os.path.join(self.path_to_station_output, filename)
        )['data_wave'][int(in_file_index)].filled()

        if channels is not None:
            if (self._check_elements(channels, self.channels) or
                    self._check_elements(channels, self.detailed_channels)):
                # Filter by channels chosen
                channel_indices = []
                for channel in channels:
                    channel_indices += [
                        index
                        for index, element in enumerate(self.detailed_channels)
                        if element.startswith(channel)]
            else:
                raise Exception('Only the following channels exist: ' +
                                ', '.join(self.detailed_channels))
        else:
            channel_indices = np.arange(len(self.detailed_channels))
        if time_limits is not None:
            if (np.min(self.data_time) < time_limits[0] and
                    np.max(self.data_time) > time_limits[1]):
                # Find the indices of elements between t_min and t_max
                indices = np.where((self.data_time >= time_limits[0]) &
                                   (self.data_time <= time_limits[1]))
            else:
                raise Exception('Times must be between: ' +
                                str(min(self.data_time)) +
                                ' and ' + str(max(self.data_time)))
        else:
            indices = np.arange(len(self.data_time))
        result = wave_data[channel_indices, indices]
        if result.ndim == 1:
            return result.reshape(1, -1)
        else:
            return result

    def stream(self, networks: list,
               station_names: list,
               locations: list = None,
               channels: list = None,
               time_limits: list = None) -> obspy.Stream:
        """Returns stream with data from the chosen stations and channels

        Args:
            networks (list): List of strings with networks
            station_names (list): List of strings with stations
            locations (list): List of strings with locations
            channels (str): String of channels
            data_type (str, optional): Type of data output (eg RTZ). Defaults
            to 'RTZ'.

        Returns:
            obspy.Stream: Obspy Stream of data
        """

        # Initialize stream that will hold all the traces
        stream = obspy.Stream()
        # Compute sampling delta (it is the same for all traces) and no of
        # points per trace
        delta = self.data_time[1] - self.data_time[0]
        npts = len(self.data_time)
        # Form full station name from components given (not including channel
        # in the name)
        for i, network in enumerate(networks):
            station_name = station_names[i]
            if locations is None:
                location = ''
            else:
                location = locations[i]

            if channels is None:
                selected_detailed_channels = self.detailed_channels
            else:
                selected_detailed_channels = [
                    element for element in self.detailed_channels
                    if any(element.startswith(prefix) for prefix in channels)]

            for chn_index, chn in enumerate(selected_detailed_channels):
                if location == '':
                    # extract data
                    try:
                        wavefield_data = self.load_data_at_station(
                            network, station_name, channels, time_limits
                            )
                    except Exception:
                        wavefield_data = self.load_data_at_station(
                            network, station_name, channels, time_limits
                            )
                    # form obspy trace
                    trace = obspy.Trace(wavefield_data[chn_index])
                    trace.stats.delta = delta
                    trace.stats.ntps = npts
                    trace.stats.network = network
                    trace.stats.station = station_name
                    trace.stats.location = location
                    trace.stats.channel = chn
                    trace.stats.starttime = self.starttime
                    stream.append(trace)
                else:
                    # extract data
                    wavefield_data = self.load_data_at_station(
                        network, station_name, channels, time_limits
                        )
                    # form obspy trace
                    trace = obspy.Trace(wavefield_data[chn_index])
                    trace.stats.delta = delta
                    trace.stats.ntps = npts
                    trace.stats.network = network
                    trace.stats.station = station_name
                    trace.stats.location = location
                    trace.stats.channel = chn
                    trace.stats.starttime = self.starttime
                    stream.append(trace)

        return stream

    def _get_time(self) -> np.ndarray:
        """Time axis of waveform

        Returns:
            np.ndarray: time
        """
        return next(iter(self._nc_data.values()))['data_time'][:].filled()

    def _check_elements(self, list1, list2):
        """Checks if all elements in list1 can be found in list2.

        Args:
            list1 (list): The first list.
            list2 (list): The second list.

        Returns:
            bool: True if all elements in list1 are found in list2, False
            otherwise.
            list: List of elements from list1 that are not found in list2.
        """
        missing_elements = [element for element in list1 if
                            element not in list2]
        if len(missing_elements) == 0:
            return True
        else:
            return False

    @property
    def starttime(self) -> UTCDateTime:
        """Get the start time of the waveform.

        Returns:
            obspy.UTCDateTime: Start time of the waveform
        """
        default_event_time = UTCDateTime("1970-01-01T00:00:00.0Z")
        return default_event_time + self.data_time[0]

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
            parent_directory, base_name = os.path.split(current_directory)
            if 'output' in base_name:
                return parent_directory
            elif current_directory == parent_directory:
                # Reached the root directory, "output" directory not found
                return None
            current_directory = parent_directory

    def _get_detailed_channels(self):
        """ The existence of this function is the terrbile
         consequence of the absence of a list of channels
          in the metadata of stations output. It's horrible """

        U = ['U1', 'U2', 'U3']
        G = ['G11', 'G12', 'G13', 'G21', 'G22', 'G23', 'G31', 'G32', 'G33']
        E = ['E11', 'E22', 'E33', 'E23', 'E13', 'E12']
        S = ['S11', 'S12', 'S13', 'S22', 'S23', 'S33']
        R = ['R1', 'R2', 'R3']

        detailed_channels = []
        for channel_type in self.channels:
            if channel_type == 'U':
                detailed_channels.extend(U)
            elif channel_type == 'G':
                detailed_channels.extend(G)
            elif channel_type == 'E':
                detailed_channels.extend(E)
            elif channel_type == 'S':
                detailed_channels.extend(S)
            elif channel_type == 'R':
                detailed_channels.extend(R)

        return detailed_channels

    def parse_to_mseed(self, channels: list = None,
                       time_limits: list = None):
        """Parse the data to MiniSEED format.

        Args:
            channels (list, optional): List of channel names. Defaults to None.
            time_limits (list, optional): List of time limits [t_min, t_max].
            Defaults to None.

        Returns:
            obspy.Stream: ObsPy Stream object containing the data
        """
        station_key_list = self._rank_list['STATION_KEY'].tolist()[1:-1]

        # Extract the string before and after the dot
        networks = [key.split(".")[0] for key in station_key_list]
        stations = [key.split(".")[1] for key in station_key_list]

        return self.stream(networks, stations)

    def get_inventory(self):
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
            source="Inventory from AxiSEM3D STATIONS file")

        # Open station file
        stations = (pd.read_csv(self._stations_file_path,
                    delim_whitespace=True,
                    header=1,
                    names=["name",
                           "network",
                           "latitude",
                           "longitude",
                           "useless",
                           "depth"],
                    comment='#'))

        # Iterate over all stations in the stations file
        for _, station in stations.iterrows():
            # Create network if not already existent
            net_exists = False
            for network in inv:
                if network.code == station['network']:
                    net_exists = True
                    net = network
            if net_exists is False:
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
            locations.append('')  # Axisem does not use locations
            channels_list.append(self.detailed_channels)

        return inv

    def obspyfy(self):
        # Create obspyfy folder if not existent already
        obspyfy_path = self.path_to_station_output + '/obspyfied'
        if not os.path.exists(obspyfy_path):
            os.mkdir(obspyfy_path)
        cat = self.catalogue
        cat.write(obspyfy_path + '/cat.xml', format='QUAKEML')

        inv = self.get_inventory()
        inv_file_name = self._stations_file_name.split('.')[0] + '_inv.xml'
        inv.write(obspyfy_path + '/' + inv_file_name, format="stationxml")

        stream = self.parse_to_mseed()
        mseed_file_name = self.station_group_name + '.mseed'
        stream.write(obspyfy_path + '/' + mseed_file_name, format="MSEED")
