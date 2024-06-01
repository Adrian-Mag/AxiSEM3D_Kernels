from ..handlers.obspy_output import ObspyfiedOutput
from ...aux.helper_functions import window_data
from ..handlers.element_output import ElementOutput
from ...core.kernels.kernel import Kernel

import matplotlib.pyplot as plt
import shutil
import os
import numpy as np
import yaml
from abc import ABC, abstractmethod
from scipy import integrate
from ruamel.yaml import YAML
import pkg_resources
import subprocess


class ObjectiveFunction(ABC):
    def __init__(self, forward_data: ElementOutput,
                 real_data: ObspyfiedOutput = None,
                 backward_data: ElementOutput = None):

        # The objective function can be initialized with only the forward data
        # and use its methods to compute the backward wavefield
        self.forward_data = forward_data
        self.backward_data = backward_data

        # Kernel
        self.kernel = None

    @abstractmethod
    def compute_backward_field(self, station: str, network: str,
                               location: str, real_channels: str,
                               window_left: float, window_right: float):
        pass

    @abstractmethod
    def _compute_adjoint_STF(self, station: str, network: str,
                             location: str, real_channels: str,
                             window_left: float, window_right: float):
        pass

    def _save_STF(self, directory, master_time, STF, channel_type):
        for channel in channel_type:
            # Save results to a text file
            filename = os.path.join(directory, channel + '.txt')
            # Combine time and data arrays column-wise
            combined_data = np.column_stack((master_time, STF[channel]))
            # Save the combined data to a text file
            np.savetxt(filename, combined_data, fmt='%.16f', delimiter='\t')

    def initialize_kernels(self, backward_data: ElementOutput = None):
        # Check if the backward_data is known
        if self.backward_data is None and backward_data is None:
            print('No backward data was provided.'
                  ' Use "compute_backward_field" to compute it.')
        elif self.backward_data is not None:
            self.kernel = Kernel(self.forward_data, self.backward_data)
        elif backward_data is not None:
            self.backward_data = backward_data
            self.kernel = Kernel(self.forward_data, self.backward_data)

    def _make_backward_directory(self):
        source_directory = self.forward_data.path_to_simulation
        destination_directory = os.path.join(
            os.path.dirname(source_directory),
            'backward_' + os.path.basename(source_directory)
            )

        self._backward_directory = destination_directory

        try:
            # Create the destination directory if not exists
            if not os.path.exists(destination_directory):
                os.makedirs(destination_directory)
            else:
                ans = input('Backward directory already exists.'
                            ' Overwrite it? (y/n): ')
                if ans == 'y':
                    shutil.rmtree(destination_directory)
                    os.makedirs(destination_directory)
                else:
                    return

            # Copy "input" subdirectory
            input_directory = os.path.join(source_directory, "input")
            if (os.path.exists(input_directory) and
                    os.path.isdir(input_directory)):
                destination_input_directory = os.path.join(
                    destination_directory, "input"
                    )
                shutil.copytree(input_directory, destination_input_directory)

            # Copy "axisem3d" file
            axisem3d_file = os.path.join(source_directory, "axisem3d")
            if os.path.exists(axisem3d_file) and os.path.isfile(axisem3d_file):
                destination_axisem3d_file = os.path.join(destination_directory,
                                                         "axisem3d")
                shutil.copy2(axisem3d_file, destination_axisem3d_file)

            print("Files copied successfully!")
        except Exception as e:
            print(f"An error occurred: {e}")


class XObjectiveFunction(ObjectiveFunction):
    def __init__(self, forward_data: ElementOutput,
                 real_data: ObspyfiedOutput = None,
                 backward_data: ElementOutput = None):
        super().__init__(forward_data, real_data, backward_data)

    def compute_backward_field(self, tau: float, receiver_point: list,
                               window: list, cores: int = None):

        # For the forward channels need to specify wether [UZ, UR, UT] or [UZ,
        # UE, UN], etc which AxiSEM3D type was used for outputting the
        # displacement. While for the real just put what is needed to select
        # the same displacement channels via the select function from obspy, eg
        # BH* if the data is in ZRT

        #  Create the necessary directory structure and
        # files
        self._make_backward_directory()

        # Compute and save adjoint source
        self._compute_adjoint_STF(tau, receiver_point,
                                  window)

        # Modify the inparam.source file
        self._change_inparam_source(receiver_point)

        # Run the backward simulation
        self._run_backward_simulation(cores)

        # Save the backward data as property of the objective object
        self.backward_data = ElementOutput(
            os.path.join(self._backward_directory, 'output/elements')
            )

    def _run_backward_simulation(self, cores):
        if cores is None:
            cores = os.cpu_count()
        command = f"mpirun -np {cores} axisem3d"

        process = subprocess.Popen(command, shell=True,
                                   cwd=self._backward_directory)
        output, error = process.communicate()

    def _change_inparam_source(self, reciever_point: list):
        yaml = YAML()
        yaml.anchor_generator = lambda *args: None
        file_path = pkg_resources.resource_filename(
            __name__, 'adjoint_source_model.yaml'
            )
        # Open the adjoint source model file
        with open(file_path, 'r') as file:
            adjoint_source = yaml.load(file)
        # Open the forward source
        with open(self.forward_data.inparam_source, 'r') as file:
            forward_source = yaml.load(file)
        adjoint_source['time_axis']['record_length'] = forward_source['time_axis']['record_length'] # noqa
        adjoint_source['time_axis']['enforced_dt'] = forward_source['time_axis']['enforced_dt'] # noqa
        adjoint_source['time_axis']['Courant_number'] = forward_source['time_axis']['Courant_number'] # noqa
        adjoint_source['time_axis']['integrator'] = forward_source['time_axis']['integrator'] # noqa

        for point in adjoint_source['list_of_sources']:
            key = list(point.keys())[0]
            point[key]['location']['latitude_longitude'][0] = reciever_point[1]
            point[key]['location']['latitude_longitude'][1] = reciever_point[2]
        with open(os.path.join(self._backward_directory,
                               'input/inparam.source.yaml'), 'w') as file:
            yaml.dump(adjoint_source, file)

    def _compute_adjoint_STF(self, tau: float,
                             receiver_point: list,
                             window):

        # Load data

        # Get the forward wavefield displacement data as a numpy array at the
        # receiver
        forward_displacement = self.forward_data.load_data(np.array(receiver_point), # noqa
                                                           channels=['UR', 'UT', 'UZ'], # noqa
                                                           in_deg=True)[0]
        # Get the time axis at the receiver. We again assume all elements have
        # the same time axis
        first_group = self.forward_data.element_groups[0]
        forward_time = self.forward_data.element_groups_info[first_group]['metadata']['data_time'] # noqa
        dt_forward = forward_time[1] - forward_time[0]
        # Window the data
        windowed_master_time, windowed_forward_data = window_data(
            forward_time, forward_displacement, window[0], window[1]
            )
        # Compute displacement time derivative
        dfwdt = np.diff(windowed_forward_data, axis=1) / (dt_forward)
        differentiated_time_axis = windowed_master_time[0:-1] - dt_forward / 2
        # Apply the t -> T-t transformation to the residue
        dfwdt = np.flip(np.array(dfwdt))
        # Apply the t -> T-t transformation to the time
        transformed_windowed_master_time = np.flip(np.max(forward_time) -
                                                   differentiated_time_axis)
        # Compute normalization factor
        mag = np.sqrt(integrate.simpson(np.sum(dfwdt * dfwdt, axis=0),
                                        dx=dt_forward))

        # Build the STF
        STF = {}
        # Get the coordinate system
        channel_type = self.forward_data.element_groups_info[first_group]['wavefields']['coordinate_frame'] # noqa
        _, axs = plt.subplots(3, 1)
        for index, channel in enumerate(channel_type):
            # Scale velocity
            new_time = np.linspace(min(transformed_windowed_master_time),
                                   max(transformed_windowed_master_time), 100)
            STF[channel] = tau * np.interp(
                new_time, transformed_windowed_master_time, dfwdt[index]
                ) / mag
            axs[index].plot(new_time, STF[channel])
            axs[index].text(1.05, 0.5, index, transform=axs[index].transAxes)
        plt.show()

        # See if you want to save the STF or not
        ans = input('Save the STF? (y/n): ')
        if ans == 'y':
            # Save residue as STF.txt file ready to be given to AxiSEM3D
            directory = os.path.join(self._backward_directory, 'input', 'STF')
            if not os.path.exists(directory):
                os.makedirs(directory)
                print("Directory created:", directory)
                self._save_STF(directory, new_time, STF, channel_type)
            else:
                print("Directory already exists:", directory)
                ans = input('Overwrite the existing data [y/n]: ')
                if ans == 'y':
                    self._save_STF(directory, new_time, STF, channel_type)


class L2ObjectiveFunction(ObjectiveFunction):
    def __init__(self, forward_data: ElementOutput,
                 real_data: ObspyfiedOutput = None,
                 backward_data: ElementOutput = None):
        super().__init__(forward_data, real_data, backward_data)

        # Source data
        self.source_depth = forward_data.source_depth
        self.source_latitude = forward_data.source_lat
        self.source_longitude = forward_data.source_lon

    def compute_backward_field(self, station: str, network: str, location: str,
                               real_channels: str, window_left: float,
                               window_right: float):

        # For the forward channels need to specify wether [UZ, UR, UT] or [UZ,
        # UE, UN], etc which AxiSEM3D type was used for outputting the
        # displacement. While for the real just put what is needed to select
        # the same displacement channels via the select function from obspy, eg
        # BH* if the data is in ZRT

        #  Create the necessary directory structure and
        # files
        self._make_backward_directory()

        # Compute and save adjoint source
        self._compute_adjoint_STF(station, network, location, real_channels,
                                  window_left, window_right)

        # Modify the inparam.source file
        input('Modify the inparam.source file manually then press enter.')

        # Run the backward simulation
        input('Run the backward simulation then press enter.')

        # Save the backward data as property of the objective object
        self.backward_data = ElementOutput(
            os.path.join(self._backward_directory, 'output/elements')
            )

    def _change_source(self):
        # NOT WORKING MUST WRITE SOURCE FILE MANUALLY
        source_file_path = os.path.join(self._backward_directory, 'input',
                                        'inparam.source.yaml')
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
                        'use_kernel_integral': 'ERF',
                        'ascii_data_file': 'STF/Z.txt',
                        'padding': 'FIRST_LAST'
                    }
                }
            }
        ]

        bw_source_data['list_of_sources'] = new_sources

        with open(source_file_path, 'w') as file:
            yaml.dump(bw_source_data, file, default_flow_style=False)

    def _compute_adjoint_STF(self, station: str, network: str,
                             location: str, real_channels: str,
                             window_left: float = None,
                             window_right: float = None):

        # get real data as stream
        stream_real_data = self.real_data.stream
        # get real data time and time step
        real_data_time = stream_real_data[0].times('timestamp')
        dt_real_data = real_data_time[1] - real_data_time[0]
        # Select the specific station data
        stream_real_data = stream_real_data.select(station=station,
                                                   network=network,
                                                   location=location,
                                                   channel=real_channels)

        # Extract the station coordinates from the inventory associated with
        # the real data
        inventory = self.real_data.inv
        inventory = inventory.select(network=network,
                                     station=station,
                                     location=location)
        station_depth = -inventory[0][0].elevation
        station_latitude = inventory[0][0].latitude
        station_longitude = inventory[0][0].longitude

        # In anticipation for the construction of the adjoint source file
        self._latitude_longitude = [station_latitude, station_longitude]
        self._depth = station_depth

        # Put the station coords in geographic spherical [rad, lat, lon] in
        # degrees
        sta_rad = self.forward_data.Earth_Radius - station_depth
        point = [sta_rad, station_latitude, station_longitude]
        # Get the forward data as a stream at that point
        stream_forward_data = self.forward_data.stream(point, channels=['U'],
                                                       coord_in_deg=True)
        # We again assume all elements have the same time axis
        first_group = next(iter(self.forward_data.element_groups_info))
        forward_time = self.forward_data.element_groups_info[first_group]['metadata']['data_time'] # noqa
        dt_forward = forward_time[1] - forward_time[0]
        channel_type = self.forward_data.element_groups_info[first_group]['wavefields']['coordinate_frame'] # noqa

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
            interpolated_real_data = np.interp(
                master_time, real_data_time,
                stream_real_data.select(channel='U' + channel)[0].data
                )

            interpolated_forward_data = np.interp(
                master_time, forward_time,
                stream_forward_data.select(channel='U' + channel)[0].data)

            if window_left is not None and window_right is not None:
                _, windowed_real_data = window_data(master_time,
                                                    interpolated_real_data,
                                                    window_left, window_right)
                windowed_master_time, windowed_forward_data = window_data(
                    master_time, interpolated_forward_data,
                    window_left, window_right
                    )
            else:
                windowed_master_time = master_time
                windowed_real_data = interpolated_real_data
                windowed_forward_data = interpolated_forward_data
            # Apply the t -> T-t transformation to the residue and multiply
            # with -1
            residue[channel] = -np.flip(np.array(windowed_forward_data) -
                                        np.array(windowed_real_data))
            axs[index].plot(windowed_master_time, residue[channel])
            axs[index].plot(windowed_master_time, windowed_forward_data,
                            color='red')
            axs[index].plot(windowed_master_time, windowed_real_data,
                            color='blue')
            axs[index].text(1.05, 0.5, index, transform=axs[index].transAxes)
        # Apply the t -> T-t transformation to the time
        transformed_windowed_master_time = np.flip(np.max(master_time) -
                                                   windowed_master_time)
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
                self._save_STF(directory, transformed_windowed_master_time,
                               STF, channel_type)
            else:
                print("Directory already exists:", directory)
                ans = input('Overwrite the existing data [y/n]: ')
                if ans == 'y':
                    self._save_STF(directory, transformed_windowed_master_time,
                                   STF, channel_type)

    def evaluate_objective_function(self, network: str,
                                    station: str, location: str,
                                    plot_residue: bool = True) -> float:
        """
        Evaluates the objective function by computing the integral of the L2
        norm of the residue over the windowed time.

        Args:
            network (str): Network identifier.
            station (str): Station identifier.
            location (str): Location identifier.
            plot_residue (bool, optional): Whether to plot the residue.
            Defaults to True.

        Returns:
            float: The integral of the L2 norm of the residue.

        Raises:
            ValueError: If invalid network, station, or location is provided.
        """

        # Load real data
        stream_real_data = self.real_data.stream.select(station=station)
        real_data_time = stream_real_data[0].times(type="relative")
        dt_real_data = real_data_time[1] - real_data_time[0]

        # Extract station coordinates from inventory
        inventory = self.real_data.inv.select(network=network, station=station,
                                              location=location)
        station_depth = -inventory[0][0].elevation
        station_latitude = inventory[0][0].latitude
        station_longitude = inventory[0][0].longitude

        # Load synthetic data
        sta_rad = self.forward_data.Earth_Radius - station_depth
        point = [sta_rad, station_latitude, station_longitude]
        stream_forward_data = self.forward_data.stream(point)
        forward_time = self.forward_data.data_time
        dt_forward = forward_time[1] - forward_time[0]
        channel_type = self.forward_data.coordinate_frame

        # Compute residue
        t_max = min(real_data_time[-1], forward_time[-1])
        t_min = max(real_data_time[0], forward_time[0])
        dt = max(dt_real_data, dt_forward)
        master_time = np.arange(t_min, t_max, dt)

        interpolated_real_data = np.vstack([
            np.interp(master_time, real_data_time,
                      stream_real_data.select(channel='U' + channel)[0].data)
            for channel in channel_type
        ])
        interpolated_forward_data = np.vstack([
            np.interp(master_time, forward_time,
                      stream_forward_data.select(channel='U' + channel)[0].data) # noqa
            for channel in channel_type
        ])

        windowed_master_time = master_time
        windowed_real_data = interpolated_real_data
        windowed_forward_data = interpolated_forward_data

        if self.window_left is not None and self.window_right is not None:
            windowed_master_time, windowed_real_data = window_data(
                master_time, interpolated_real_data,
                self.window_left, self.window_right
                )
            _, windowed_forward_data = window_data(
                master_time, interpolated_forward_data,
                self.window_left, self.window_right
                )

        residue = windowed_forward_data - windowed_real_data
        residue_norm = np.linalg.norm(residue, axis=0)
        integral = np.trapz(residue_norm, x=windowed_master_time)

        if plot_residue:
            # Plot windowed_real_data, windowed_forward_data, residue, and L2
            # norm of residue
            fig, ax = plt.subplots(len(channel_type), 1, figsize=(10, 6))
            fig.suptitle('Windowed Real Data, Windowed Forward Data,'
                         ' and Residue')

            for i, channel in enumerate(channel_type):
                ax[i].plot(windowed_master_time, windowed_real_data[i],
                           label='Windowed Real Data')
                ax[i].plot(windowed_master_time, windowed_forward_data[i],
                           label='Windowed Forward Data')
                ax[i].plot(windowed_master_time, residue[i], label='Residue')
                ax[i].set_xlabel('Time')
                ax[i].set_ylabel('Amplitude')
                ax[i].legend()

            plt.show()

        return 0.5 * integral
