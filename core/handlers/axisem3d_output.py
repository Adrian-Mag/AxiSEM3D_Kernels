import os
import yaml
from obspy.core.event import Catalog, Event, Origin, FocalMechanism, MomentTensor, Tensor
from obspy import UTCDateTime
from obspy.geodetics import FlinnEngdahl
import fnmatch
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
        outputs (dict): Dictionary containing information about the simulation outputs.
        simulation_name (str): Name of the simulation.
        Earth_Radius (int): Radius of the Earth in meters.

    Methods:
        _find_catalogue(): Find the catalogue file.
        _find_outputs(): Find the output directories.
        _search_files(directory, keyword, include_subdirectories=False): Search for files containing a specific keyword.
        catalogue(): Get the simulation catalogue.

    """

    def __init__(self, path_to_simulation):
        """
        Initialize the AxiSEM3DOutput instance.

        Args:
            path_to_simulation (str): Path to the AxiSEM3D simulation directory.
        """

        # Info about the structure of the output files
        self.path_to_simulation = path_to_simulation
        self.inparam_model = self.path_to_simulation + '/input/inparam.model.yaml'
        self.inparam_nr = self.path_to_simulation + '/input/inparam.nr.yaml'
        self.inparam_output = self.path_to_simulation + '/input/inparam.output.yaml'
        self.inparam_source = self.path_to_simulation + '/input/inparam.source.yaml'
        self.inparam_advanced = self.path_to_simulation + '/input/inparam.advanced.yaml'
        self.outputs = self._find_outputs()
        self.simulation_name = os.path.basename(self.path_to_simulation)

        # Info about the source
        self._catalogue = self._find_catalogue()
        
        # Info about model
        self.Earth_Radius = 6371000 # m


    def _find_catalogue(self):
        """
        Find the catalogue file.

        Returns:
            obspy.core.event.Catalog or None: Catalog object if a single catalogue file is found, otherwise None.
        """
        catalogues = glob.glob(os.path.join(self.path_to_simulation, 'input', '*cat.xml'))
        if len(catalogues) == 1:
            return read_events(catalogues[0])
        elif len(catalogues) == 0:
            return None
        else:
            print('Multiple catalogues were found, therefore we abort.')
            return None


    def _find_outputs(self):
        """
        Find the output directories.

        Returns:
            dict: Dictionary containing information about the simulation outputs.
        """
        outputs = {'elements': {}, 'stations': {}}

        for output_type in ['elements', 'stations']:
            path_to_output = os.path.join(self.path_to_simulation, 'output', output_type)
            output_dirs = glob.glob(os.path.join(path_to_output, '*'))
            
            for output_dir in output_dirs:
                output_name = os.path.basename(output_dir)
                outputs[output_type][output_name] = {'path': output_dir, 'obspyfied': {}}
                obspyfied_path = os.path.join(output_dir, 'obspyfied')
                
                if os.path.exists(obspyfied_path):
                    mseed_files = glob.glob(os.path.join(obspyfied_path, '*.mseed'))
                    if len(mseed_files) == 0:
                        mseed_files = None
                    inv_files = glob.glob(os.path.join(obspyfied_path, '*inv.xml'))
                    if len(inv_files) == 0:
                        inv_files = None
                    outputs[output_type][output_name]['obspyfied'] = {'path': obspyfied_path, 'mseed': mseed_files, 'inventory': inv_files}
                else:
                    obspyfied_path = None
                    outputs[output_type][output_name]['obspyfied'] = None
        return outputs


    @property
    def catalogue(self):
        """
        Get the simulation catalogue.

        Returns:
            obspy.core.event.Catalog: Catalog object representing the simulation catalogue.
        """
        if self._catalogue is None:
            with open(self.inparam_source, 'r') as file:
                    source_yaml = yaml.load(file, Loader=yaml.FullLoader)
                    cat = Catalog()
                    for source in source_yaml['list_of_sources']:
                        for items in source.items():
                            event = Event()
                            origin = Origin()

                            origin.time = UTCDateTime("1970-01-01T00:00:00.0Z") # default in obspy
                            origin.latitude = items[1]['location']['latitude_longitude'][0]
                            origin.longitude = items[1]['location']['latitude_longitude'][1]
                            origin.depth = items[1]['location']['depth']
                            origin.depth_type = "operator assigned"
                            origin.evaluation_mode = "manual"
                            origin.evaluation_status = "preliminary"
                            origin.region = FlinnEngdahl().get_region(origin.longitude, origin.latitude)

                            if items[1]['mechanism']['type'] == 'FORCE_VECTOR':
                                m_rr = items[1]['mechanism']['data'][0]
                                m_tt = items[1]['mechanism']['data'][1]
                                m_pp = items[1]['mechanism']['data'][2]
                                m_rt = 0
                                m_rp = 0
                                m_tp = 0
                            elif items[1]['mechanism']['type'] == 'FLUID_PRESSURE':
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
            cat.write(self.path_to_simulation + '/input/' + self.simulation_name + '_cat.xml', format='QUAKEML')
            self._catalogue = cat
            return cat
        else: 
            return self._catalogue