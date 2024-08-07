import os
import yaml
from obspy.core.event import Catalog, Event, Origin, FocalMechanism, MomentTensor, Tensor # noqa
from obspy import UTCDateTime
from obspy.geodetics import FlinnEngdahl
from obspy import read_events
import glob


class AxiSEM3DOutput:
    """
    A class representing AxiSEM3D simulation output.

    Attributes:
        path_to_simulation (str): Path to the AxiSEM3D simulation directory.
        inparam_model (str): Path to the inparam.model.yaml file.
        inparam_nr (str): Path to the inparam.nr.yaml file.
        inparam_output (str): Path to the inparam.output.yaml file.
        inparam_source (str): Path to the inparam.source.yaml file.
        inparam_advanced (str): Path to the inparam.advanced.yaml file.
        outputs (dict): Dictionary containing information about the simulation
        outputs.
        simulation_name (str): Name of the simulation.
        Domain_Radius (int): Radius of the Earth in meters.
        base_model (dict): Dictionary containing the base model data.

    Methods:
        _find_catalogue(): Find the catalogue file.
        _find_outputs(): Find the output directories.
        _search_files(directory, keyword, include_subdirectories=False): Search
        for files containing a specific keyword.
        catalogue(): Get the simulation catalogue.

    """

    def __init__(self, path_to_simulation, path_to_base_model: str = None):
        """
        Initialize the AxiSEM3DOutput instance.

        Args:
            path_to_simulation (str): Path to the AxiSEM3D simulation
            directory.
        """
        self.path_to_simulation = path_to_simulation
        # Info about the input file paths
        self.inparam_model = self.path_to_simulation + '/input/inparam.model.yaml' # noqa
        self.inparam_nr = self.path_to_simulation + '/input/inparam.nr.yaml'
        self.inparam_output = self.path_to_simulation + '/input/inparam.output.yaml' # noqa
        self.inparam_source = self.path_to_simulation + '/input/inparam.source.yaml' # noqa
        self.inparam_advanced = self.path_to_simulation + '/input/inparam.advanced.yaml' # noqa
        # Info about the structure of the output files
        self.outputs = self._find_outputs()
        # We give a name to the simulation
        self.simulation_name = os.path.basename(self.path_to_simulation)
        # Info about the source
        self._stored_catalogue = self._find_catalogue()[0]
        # Info about model (currently only for global models)
        if path_to_base_model is None:
            # We search for a bm file in the input folder
            bm_files = glob.glob(os.path.join(self.path_to_simulation, 'input', '*.bm')) # noqa
            # If there are multiple bm files, we take the first one
            if len(bm_files) > 1:
                print('Multiple bm files were found, we take the first one.')
            elif len(bm_files) == 0:
                raise ValueError('No bm files were found.')
            path_to_base_model = bm_files[0]
        # After finding the base model, we read and save its contents (they
        # must be easily accessible from this class for later)
        self.base_model = self._read_model_file(path_to_base_model)
        # Now we determine the Earth radius. We save it as an attribute of the
        # class rather than as a key in the base_model dictionary.
        if 'axisem3d' in os.path.basename(path_to_base_model):
            self.Domain_Radius = self.base_model['DISCONTINUITIES'][0]
        else:
            self.Domain_Radius = max(self.base_model['DATA']['radius'])
        with open(self.inparam_model, 'r') as file:
            model_yaml = yaml.load(file, Loader=yaml.FullLoader)
            # Check for any 3D models
            if len(model_yaml['list_of_3D_models']) == 0:
                self.threeD_models = None
            else:
                pass

    def _read_model_file(self, file_path):
        """
        Read the model file.

        Args:
            file_path (str): Path to the model file.

        Returns:
            dict: Dictionary containing the model data.
        """
        if 'axisem3d' in os.path.basename(file_path):
            # The bm file is of axisem3d type
            data = {}
            data['type'] = 'axisem3d'
            with open(file_path, 'r') as file:
                lines = file.readlines()

            possible_model_properties = ['RHO', 'VP', 'VS', 'QKAPPA', 'QMU']
            unit_dependent_model_properties = ['RHO', 'VP', 'VS']
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if not line.startswith('#') and line:
                    key, *values = line.split()
                    if key in ['NAME', 'MODEL_TYPE', 'ANELASTIC',
                               'ANISOTROPIC', 'UNITS']:
                        data[key] = values[0]
                    elif key in ['COLUMNS']:
                        data[key] = values
                    elif key in ['DISCONTINUITIES']:
                        data[key] = [float(value) for value in values]
                    elif key in ['REFERENCE_FREQUENCY', 'NREGIONS',
                                 'MAX_POLY_DEG', 'SCALE']:
                        data[key] = float(values[0])
                    elif key in possible_model_properties:
                        j = i + 1
                        values = []
                        while j < len(lines):
                            line = lines[j].strip()
                            if line.startswith('#') or not line:
                                break
                            values.append(float(line))
                            j += 1
                        data[key] = values
                i += 1
            for key in unit_dependent_model_properties:
                data[key] = [element * 1e3 for element in data[key]]
            data['DISCONTINUITIES'] = [element * 1e3 for element in data['DISCONTINUITIES']] # noqa
            data['R'] = data['DISCONTINUITIES']
        else:
            # The bm file is of axisem type
            data = {}
            data['type'] = 'axisem'
            with open(file_path, 'r') as file:
                lines = file.readlines()

            create_data_lists = True
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if not line.startswith('#') and line:
                    key, *values = line.split()
                    if key in ['NAME', 'ANELASTIC', 'ANISOTROPIC', 'UNITS']:
                        data[key] = values[0]
                    elif key in ['COLUMNS']:
                        data[key] = values
                    else:
                        if create_data_lists:
                            create_data_lists = False
                            data['DATA'] = {}
                            for key in data['COLUMNS']:
                                data['DATA'][key] = []
                        for index, key in enumerate(data['COLUMNS']):
                            data['DATA'][key].append(float(line.split()[index])) # noqa
                i += 1
            data['DISCONTINUITIES'] = [data['DATA']['radius'][0]]
            for i in range(len(data['DATA']['radius']) - 1):
                if data['DATA']['radius'][i] == data['DATA']['radius'][i+1]:
                    data['DISCONTINUITIES'].append(data['DATA']['radius'][i])
            data['DISCONTINUITIES'].append(data['DATA']['radius'][-1])
        return data

    def _find_catalogue(self):
        """
        Find the catalogue file.

        Returns:
            obspy.core.event.Catalog or None: Catalog object if a single
            catalogue file is found, otherwise None.
        """
        catalogues = glob.glob(os.path.join(self.path_to_simulation, 'input', '*cat*.xml')) # noqa
        if len(catalogues) == 1:
            return read_events(catalogues[0])
        elif len(catalogues) == 0:
            print('No catalogues were found.')
            return (None, 1)
        else:
            print('Multiple catalogues were found, therefore we abort.')
            return (None, 2)

    def _find_outputs(self):
        """
        Find the output directories.

        Returns:
            dict: Dictionary containing information about the simulation
            outputs.
        """
        outputs = {'elements': {}, 'stations': {}}

        for output_type in ['elements', 'stations']:
            path_to_output = os.path.join(self.path_to_simulation, 'output',
                                          output_type)
            output_dirs = glob.glob(os.path.join(path_to_output, '*'))

            for output_dir in output_dirs:
                output_name = os.path.basename(output_dir)
                obspyfied_path = os.path.join(output_dir, 'obspyfied')

                obspyfied_data = None
                if os.path.exists(obspyfied_path):
                    mseed_files = glob.glob(os.path.join(obspyfied_path,
                                                         '*.mseed'))
                    inv_files = glob.glob(os.path.join(obspyfied_path,
                                                       '*inv.xml'))

                    mseed_files = None if len(mseed_files) == 0 else mseed_files # noqa
                    inv_files = None if len(inv_files) == 0 else inv_files

                    obspyfied_data = {'path': obspyfied_path,
                                      'mseed': mseed_files,
                                      'inventory': inv_files}

                outputs[output_type][output_name] = {'path': output_dir,
                                                     'obspyfied': obspyfied_data} # noqa

        return outputs

    @property
    def catalogue(self):
        """
        Get the simulation catalogue.

        Returns:
            obspy.core.event.Catalog: Catalog object representing the
            simulation catalogue.
        """
        if self._stored_catalogue is None:
            with open(self.inparam_source, 'r') as file:
                source_yaml = yaml.load(file, Loader=yaml.FullLoader)
                cat = Catalog()
                for source in source_yaml['list_of_sources']:
                    for items in source.items():
                        event = Event()
                        origin = Origin()

                        origin.time = UTCDateTime("1970-01-01T00:00:00.0Z") # default in obspy # noqa
                        origin.latitude = items[1]['location']['latitude_longitude'][0] # noqa
                        origin.longitude = items[1]['location']['latitude_longitude'][1] # noqa
                        origin.depth = items[1]['location']['depth']
                        origin.depth_type = "operator assigned"
                        origin.evaluation_mode = "manual"
                        origin.evaluation_status = "preliminary"
                        origin.region = FlinnEngdahl().get_region(origin.longitude, origin.latitude) # noqa

                        if items[1]['mechanism']['type'] == 'FORCE_VECTOR':
                            m_rr = items[1]['mechanism']['data'][0]
                            m_tt = items[1]['mechanism']['data'][1]
                            m_pp = items[1]['mechanism']['data'][2]
                            m_rt = 0
                            m_rp = 0
                            m_tp = 0
                        elif items[1]['mechanism']['type'] == 'FLUID_PRESSURE': # noqa
                            m_rr = items[1]['mechanism']['data'][0]
                            m_tt = 0
                            m_pp = 0
                            m_rt = 0
                            m_rp = 0
                            m_tp = 0
                        else:
                            m_rr = items[1]['mechanism']['data'][0]
                            m_tt = items[1]['mechanism']['data'][1]
                            m_pp = items[1]['mechanism']['data'][2]
                            m_rt = items[1]['mechanism']['data'][3]
                            m_rp = items[1]['mechanism']['data'][4]
                            m_tp = items[1]['mechanism']['data'][5]

                        focal_mechanisms = FocalMechanism()
                        tensor = Tensor()
                        moment_tensor = MomentTensor()
                        tensor.m_rr = m_rr
                        tensor.m_tt = m_tt
                        tensor.m_pp = m_pp
                        tensor.m_rt = m_rt
                        tensor.m_rp = m_rp
                        tensor.m_tp = m_tp
                        moment_tensor.tensor = tensor
                        focal_mechanisms.moment_tensor = moment_tensor

                        cat.append(event)
                        event.origins = [origin]
                        event.focal_mechanisms = [focal_mechanisms]
            cat.write(self.path_to_simulation + '/input/' +
                      self.simulation_name + '_cat.xml', format='QUAKEML')
            self._stored_catalogue = cat
            return cat
        else:
            return self._stored_catalogue
