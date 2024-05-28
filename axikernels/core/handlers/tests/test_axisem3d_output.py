import unittest
import os
from axisem3d_output.core.handlers.axisem3d_output import AxiSEM3DOutput
from obspy.core.event import Catalog
import glob 

class AxiSEM3DOutputTestCase(unittest.TestCase):
    def setUp(self):
        self.path_to_simulation = "core/handlers/tests/NORMAL_FAULT_100KM"
        self.output = AxiSEM3DOutput(self.path_to_simulation)


    def test_attributes(self):
        self.assertEqual(self.output.path_to_simulation, self.path_to_simulation)
        self.assertEqual(self.output.inparam_model, os.path.join(self.path_to_simulation, "input/inparam.model.yaml"))
        self.assertEqual(self.output.inparam_nr, os.path.join(self.path_to_simulation, "input/inparam.nr.yaml"))
        self.assertEqual(self.output.inparam_output, os.path.join(self.path_to_simulation, "input/inparam.output.yaml"))
        self.assertEqual(self.output.inparam_source, os.path.join(self.path_to_simulation, "input/inparam.source.yaml"))
        self.assertEqual(self.output.inparam_advanced, os.path.join(self.path_to_simulation, "input/inparam.advanced.yaml"))
        self.assertEqual(self.output.simulation_name, os.path.basename(self.path_to_simulation))
        self.assertEqual(self.output.Earth_Radius, 6371000)        


    def test_find_catalogue_single_file(self):
        # Create a catalogue file if it does not exist yet
        self.output.catalogue
        catalog = self.output._find_catalogue()
        self.assertIsInstance(catalog, Catalog)
        self.assertEqual(len(catalog), 1)
        self._remove_catalogues()


    def test_find_catalogue_no_file(self):
        self._remove_catalogues()
        catalog = self.output._find_catalogue()
        self.assertIsNone(catalog[0])  # No catalogue file should return None
        self.assertEqual(catalog[1], 1)
        self._remove_catalogues()


    def test_find_catalogue_multiple_files(self):
        # Create multiple catalog files
        self._remove_catalogues()
        catalog_path1 = os.path.join(self.path_to_simulation, "input", "cat1.xml")
        catalog_path2 = os.path.join(self.path_to_simulation, "input", "cat2.xml")
        with open(catalog_path1, "w") as file:
            file.write("<event>Catalog1</event>")
        with open(catalog_path2, "w") as file:
            file.write("<event>Catalog2</event>")

        catalog = self.output._find_catalogue()
        self.assertIsNone(catalog[0])  # Multiple catalogues should return None
        self.assertEqual(catalog[1], 2)
        self._remove_catalogues()


    def test_find_outputs_no_obspyfy(self):
        outputs = self.output._find_outputs()

        # Assert the structure of the outputs dictionary
        self.assertIsInstance(outputs, dict)
        self.assertIn('elements', outputs)
        self.assertIn('stations', outputs)

        elements = outputs['elements']
        stations = outputs['stations']

        # Assert the elements directory
        self.assertIsInstance(elements, dict)
        self.assertDictEqual(elements, {})  # Expected empty dictionary

        self.assertIsInstance(stations, dict)

        # Assert the stations directory without obspyfy
        self.assertIn('Station_grid', stations)
        station_grid = stations['Station_grid']
        self.assertIsInstance(station_grid, dict)
        self.assertIn('path', station_grid)
        self.assertIn('obspyfied', station_grid)
        station_path = station_grid['path']
        station_obspyfied = station_grid['obspyfied']
        self.assertEqual(station_path, os.path.join(self.path_to_simulation, 'output', 'stations', 'Station_grid'))
        self.assertIsNone(station_obspyfied)

        # Assert the one with obspyfy but empty
        self.assertIn('with_obspyfy_empty', stations)
        station_grid = stations['with_obspyfy_empty']
        self.assertIn('path', station_grid)
        self.assertIn('obspyfied', station_grid)
        station_path = station_grid['path']
        station_obspyfied = station_grid['obspyfied']
        self.assertEqual(station_path, os.path.join(self.path_to_simulation, 'output', 'stations', 'with_obspyfy_empty'))
        self.assertIsInstance(station_obspyfied, dict)
        self.assertEqual(station_obspyfied['path'], os.path.join(self.path_to_simulation, 'output', 'stations', 'with_obspyfy_empty', 'obspyfied'))
        self.assertIsNone(station_obspyfied['mseed'])
        self.assertIsNone(station_obspyfied['inventory'])

        # Assert the one with 
        self.assertIn('with_obspyfy', stations)
        station_grid = stations['with_obspyfy']
        self.assertIn('path', station_grid)
        self.assertIn('obspyfied', station_grid)
        station_path = station_grid['path']
        station_obspyfied = station_grid['obspyfied']
        self.assertEqual(station_path, os.path.join(self.path_to_simulation, 'output', 'stations', 'with_obspyfy'))
        self.assertIsInstance(station_obspyfied, dict)
        self.assertEqual(station_obspyfied['path'], os.path.join(self.path_to_simulation, 'output', 'stations', 'with_obspyfy', 'obspyfied'))
        self.assertEqual(station_obspyfied['mseed'], glob.glob(os.path.join(station_obspyfied['path'], '*.mseed')))
        self.assertEqual(station_obspyfied['inventory'], glob.glob(os.path.join(station_obspyfied['path'], '*inv.xml')))

    def _remove_catalogues(self):
        files = glob.glob(os.path.join(self.path_to_simulation, 'input', "*cat*.xml"))

        for file in files:
            os.remove(file)