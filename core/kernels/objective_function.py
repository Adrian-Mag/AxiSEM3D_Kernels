from ..handlers.station_output import StationOutput
from ..handlers.obspy_output import ObspyfiedOutput
from ...aux.helper_functions import window_data
from ..handlers.element_output import ElementOutput

from ..handlers.element_output import ElementOutput
import matplotlib.pyplot as plt
import shutil
import os 
import numpy as np
import yaml

class L2Objective_Function:
    def __init__(self, forward_data:ElementOutput, real_data:ObspyfiedOutput):
        self.forward_data_obj = forward_data
        self.real_data_obj = real_data

        # Source data
        self.source_depth = forward_data.source_depth
        self.source_latitude = forward_data.source_lat
        self.source_longitude = forward_data.source_lon


    def _make_backward_directory(self):
        source_directory = self.forward_data_obj.path_to_simulation
        destination_directory = os.path.join(os.path.dirname(source_directory), 
                                           'backward_' + os.path.basename(source_directory))
        self._backward_directory = destination_directory
        try:
            # Create the destination directory if not exists
            if not os.path.exists(destination_directory):
                os.makedirs(destination_directory)
            else: 
                ans = input('Backward directory already exists. Overwrite it? (y/n): ')
                if ans == 'y':
                    shutil.rmtree(destination_directory)
                    os.makedirs(destination_directory)

            # Copy "input" subdirectory
            input_directory = os.path.join(source_directory, "input")
            if os.path.exists(input_directory) and os.path.isdir(input_directory):
                destination_input_directory = os.path.join(destination_directory, "input")
                shutil.copytree(input_directory, destination_input_directory)

            # Copy "axisem3d" file
            axisem3d_file = os.path.join(source_directory, "axisem3d")
            if os.path.exists(axisem3d_file) and os.path.isfile(axisem3d_file):
                destination_axisem3d_file = os.path.join(destination_directory, "axisem3d")
                shutil.copy2(axisem3d_file, destination_axisem3d_file)

            print("Files copied successfully!")
        except Exception as e:
            print(f"An error occurred: {e}")


    def compute_backward_field(self, station: str, network: str, location: str, real_channels: str,
                               window_left: float, window_right: float):
        
        # For the forward channels need to specify wether [UZ, UR, UT] or [UZ,
        # UE, UN], etc which AxiSEM3D type was used for outputting the
        # displacement. While for the real just put what is needed to select the
        # same displacement channels via the select function from obspy, eg BH*
        # if the data is in ZRT

        #  Create the necessary directory structure and
        # files
        self._make_backward_directory()

        # Compute and save adjoint source
        self._compute_adjoint_STF(station, network, location, real_channels,
                                  window_left, window_right)
        
        # Modify the inparam.source file 
        input('Modify the inparam.source file manually then press enter.')


    def _change_source(self):
        # NOT WORKING MUST WRITE SOURCE FILE MANUALLY
        source_file_path = os.path.join(self._backward_directory, 'input', 'inparam.source.yaml')
        with open(source_file_path, "r") as file:
            bw_source_data = yaml.safe_load(file)

        # Remove all but the first source in the inparam.source file
        first_source = bw_source_data['list_of_sources'][0]

        # Create new entries for Z, R, and T
        new_sources = [
            {
                'Z': {
                    'location': {
                        'latitude_longitude': self._latitude_longitude,
                        'depth': self._depth,
                        'ellipticity': False,
                        'depth_below_solid_surface': True,
                        'undulated_geometry': True
                    },
                    'mechanism': {
                        'type': 'FORCE_VECTOR',
                        'data': [1, 0, 0],
                        'unit': 1
                    },
                    'source_time_function': {
                        'class_name': 'StreamSTF',
                        'half_duration': 25,
                        'decay_factor': 1.628,
                        'time_shift': 0.000e+00,
                        'use_derivative_integral': 'ERF',
                        'ascii_data_file': 'STF/Z.txt',
                        'padding': 'FIRST_LAST'
                    }
                }
            }
        ]

        bw_source_data['list_of_sources'] = new_sources

        with open(source_file_path, 'w') as file:
            yaml.dump(bw_source_data, file, default_flow_style=False)


    def _compute_adjoint_STF(self, 
                             station: str, network: str, location:str, 
                             real_channels: str,
                             window_left: float=None, 
                             window_right: float=None):

        # get real data as stream
        stream_real_data = self.real_data_obj.stream
        # get real data time and time step
        real_data_time = stream_real_data[0].times('timestamp')
        dt_real_data = real_data_time[1] - real_data_time[0]
        # Select the specific station data
        stream_real_data = stream_real_data.select(station=station, 
                                                   network=network, 
                                                   location=location, 
                                                   channel=real_channels)

        # Extract the station coordinates from the inventory associated with the
        # real data
        inventory = self.real_data_obj.inv
        inventory = inventory.select(network=network, 
                                    station=station, 
                                    location=location)
        station_depth = -inventory[0][0].elevation
        station_latitude = inventory[0][0].latitude
        station_longitude = inventory[0][0].longitude  

        # In anticipation for the construction of the adjoint source file
        self._latitude_longitude = [station_latitude, station_longitude]
        self._depth = station_depth  

        # Put the station coords in geographic spherical [rad, lat, lon] in degrees
        sta_rad = self.forward_data_obj.Earth_Radius - station_depth
        point = [sta_rad, station_latitude, station_longitude]
        # Get the forward data as a stream at that point
        stream_forward_data = self.forward_data_obj.stream(point, channels=['U'], coord_in_deg=True)
        # We again assume all elements have the same time axis
        first_group = next(iter(self.forward_data_obj.element_groups_info))
        forward_time = self.forward_data_obj.element_groups_info[first_group]['metadata']['data_time']
        dt_forward = forward_time[1] - forward_time[0]
        channel_type = self.forward_data_obj.element_groups_info[first_group]['wavefields']['coordinate_frame']

        # Find the master time (minmax/maxmin)
        t_max = min(real_data_time[-1], forward_time[-1])
        t_min = max(real_data_time[0], forward_time[0])
        dt = max(dt_real_data, dt_forward)
        master_time = np.arange(t_min, t_max, dt)

        # Project both arrays on the master time
        interpolated_real_data = []
        interpolated_forward_data = []
        residue = {}

        fig, axs = plt.subplots(3, 1)
        for index, channel in enumerate(channel_type):
            interpolated_real_data = np.interp(master_time, 
                                            real_data_time, 
                                            stream_real_data.select(channel='U' + channel)[0].data)
            interpolated_forward_data = np.interp(master_time, 
                                                forward_time, 
                                                stream_forward_data.select(channel='U' + channel)[0].data)
            if window_left is not None and window_right is not None:
                _, windowed_real_data = window_data(master_time, interpolated_real_data, window_left, window_right)
                windowed_master_time, windowed_forward_data = window_data(master_time, interpolated_forward_data, 
                                                                    window_left, window_right)
            else:
                windowed_master_time = master_time
                windowed_real_data = interpolated_real_data
                windowed_forward_data = interpolated_forward_data
            # Apply the t -> T-t transformation to the dresidue and multiply with -1
            residue[channel] = -np.flip(np.array(windowed_forward_data) - np.array(windowed_real_data))
            axs[index].plot(windowed_master_time, residue[channel])
            axs[index].plot(windowed_master_time, windowed_forward_data, color='red')
            axs[index].plot(windowed_master_time, windowed_real_data, color='blue')
            axs[index].text(1.05, 0.5, index, transform=axs[index].transAxes)
        # Apply the t -> T-t transformation to the time 
        transformed_windowed_master_time = np.flip(np.max(master_time) - windowed_master_time)
        STF = residue

        # Plot
        plt.show()

        ans = input('Save the STF? (y/n): ')
        if ans == 'y':
            # Save residue as STF.txt file ready to be given to AxiSEM3D
            directory = os.path.join(self._backward_directory, 'input', 'STF')
            if not os.path.exists(directory):
                os.makedirs(directory)
                print("Directory created:", directory)
                self._save_STF(directory, transformed_windowed_master_time, STF, channel_type)
            else:
                print("Directory already exists:", directory)
                ans = input('Overwrite the existing data [y/n]: ')
                if ans == 'y':
                    self._save_STF(directory, transformed_windowed_master_time, STF, channel_type)


    def _save_STF(self, directory, master_time, STF, channel_type):
        for channel in channel_type:
            # Save results to a text file
            filename = os.path.join(directory, channel + '.txt')
            # Combine time and data arrays column-wise
            combined_data = np.column_stack((master_time, STF[channel]))
            print(STF[channel])
            # Save the combined data to a text file
            np.savetxt(filename, combined_data, fmt='%.16f', delimiter='\t')


    def evaluate_objective_function(self, network: str, station: str, location: str,
                                plot_residue: bool=True) -> float:
        """
        Evaluates the objective function by computing the integral of the L2 norm of the residue over the windowed time.

        Args:
            network (str): Network identifier.
            station (str): Station identifier.
            location (str): Location identifier.
            plot_residue (bool, optional): Whether to plot the residue. Defaults to True.

        Returns:
            float: The integral of the L2 norm of the residue.

        Raises:
            ValueError: If invalid network, station, or location is provided.
        """

        # Load real data
        stream_real_data = self.real_data_obj.stream.select(station=station)
        real_data_time = stream_real_data[0].times(type="relative")
        dt_real_data = real_data_time[1] - real_data_time[0]

        # Extract station coordinates from inventory
        inventory = self.real_data_obj.inv.select(network=network, station=station, location=location)
        station_depth = -inventory[0][0].elevation
        station_latitude = inventory[0][0].latitude
        station_longitude = inventory[0][0].longitude

        # Load synthetic data
        sta_rad = self.forward_data_obj.Earth_Radius - station_depth
        point = [sta_rad, station_latitude, station_longitude]
        stream_forward_data = self.forward_data_obj.stream(point)
        forward_time = self.forward_data_obj.data_time
        dt_forward = forward_time[1] - forward_time[0]
        channel_type = self.forward_data_obj.coordinate_frame

        # Compute residue
        t_max = min(real_data_time[-1], forward_time[-1])
        t_min = max(real_data_time[0], forward_time[0])
        dt = max(dt_real_data, dt_forward)
        master_time = np.arange(t_min, t_max, dt)

        interpolated_real_data = np.vstack([
            np.interp(master_time, real_data_time, stream_real_data.select(channel='U' + channel)[0].data)
            for channel in channel_type
        ])
        interpolated_forward_data = np.vstack([
            np.interp(master_time, forward_time, stream_forward_data.select(channel='U' + channel)[0].data)
            for channel in channel_type
        ])

        windowed_master_time = master_time
        windowed_real_data = interpolated_real_data
        windowed_forward_data = interpolated_forward_data

        if self.window_left is not None and self.window_right is not None:
            windowed_master_time, windowed_real_data = window_data(master_time, interpolated_real_data, self.window_left, self.window_right)
            _, windowed_forward_data = window_data(master_time, interpolated_forward_data, self.window_left, self.window_right)

        residue = windowed_forward_data - windowed_real_data
        residue_norm = np.linalg.norm(residue, axis=0)
        integral = np.trapz(residue_norm, x=windowed_master_time)

        if plot_residue:
            # Plot windowed_real_data, windowed_forward_data, residue, and L2 norm of residue
            fig, ax = plt.subplots(len(channel_type), 1, figsize=(10, 6))
            fig.suptitle('Windowed Real Data, Windowed Forward Data, and Residue')

            for i, channel in enumerate(channel_type):
                ax[i].plot(windowed_master_time, windowed_real_data[i], label='Windowed Real Data')
                ax[i].plot(windowed_master_time, windowed_forward_data[i], label='Windowed Forward Data')
                ax[i].plot(windowed_master_time, residue[i], label='Residue')
                ax[i].set_xlabel('Time')
                ax[i].set_ylabel('Amplitude')
                ax[i].legend()

            plt.show()

        return 0.5 * integral
