import shutil
import unittest
import os
from axikernels.core.handlers.axisem3d_output import AxiSEM3DOutput
from obspy.core.event import Catalog
import glob

# Absolute path to the committed fixture directory (works regardless of cwd)
_HERE = os.path.dirname(os.path.abspath(__file__))
_FIXTURE = os.path.join(_HERE, "NORMAL_FAULT_100KM")

# The .bm model file lives in the examples tree; copy it into the fixture
# during test-class setup so AxiSEM3DOutput can initialise.
_PKG_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
_BM_SRC = os.path.join(
    _PKG_ROOT, "examples", "data", "1D_KERNEL_EXAMPLE",
    "input", "prem_iso_elastic.bm",
)
_BM_DST = os.path.join(_FIXTURE, "input", "prem_iso_elastic.bm")


class AxiSEM3DOutputTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Copy the .bm model file into the fixture if it is not yet present."""
        if not os.path.exists(_BM_DST):
            shutil.copy(_BM_SRC, _BM_DST)

    @classmethod
    def tearDownClass(cls):
        """Remove the transient .bm file so the fixture stays lean."""
        if os.path.exists(_BM_DST):
            os.remove(_BM_DST)

    def setUp(self):
        self.path_to_simulation = _FIXTURE
        self.output = AxiSEM3DOutput(self.path_to_simulation)

    def test_attributes(self):
        self.assertEqual(self.output.path_to_simulation, self.path_to_simulation)
        self.assertEqual(
            self.output.inparam_model,
            os.path.join(self.path_to_simulation, "input/inparam.model.yaml"),
        )
        self.assertEqual(
            self.output.inparam_nr,
            os.path.join(self.path_to_simulation, "input/inparam.nr.yaml"),
        )
        self.assertEqual(
            self.output.inparam_output,
            os.path.join(self.path_to_simulation, "input/inparam.output.yaml"),
        )
        self.assertEqual(
            self.output.inparam_source,
            os.path.join(self.path_to_simulation, "input/inparam.source.yaml"),
        )
        self.assertEqual(
            self.output.inparam_advanced,
            os.path.join(self.path_to_simulation, "input/inparam.advanced.yaml"),
        )
        self.assertEqual(
            self.output.simulation_name, os.path.basename(self.path_to_simulation)
        )
        # Attribute is Domain_Radius (not Earth_Radius)
        self.assertAlmostEqual(self.output.Domain_Radius, 6371000, delta=1000)

    def test_find_catalogue_single_file(self):
        # Access the property so it creates and writes the catalogue XML.
        self.output.catalogue
        catalog = self.output._find_catalogue()
        self.assertIsInstance(catalog, Catalog)
        self.assertEqual(len(catalog), 1)
        self._remove_catalogues()

    def test_find_catalogue_no_file(self):
        self._remove_catalogues()
        catalog = self.output._find_catalogue()
        # Current implementation returns None when no catalogues are found.
        self.assertIsNone(catalog)
        self._remove_catalogues()

    def test_find_catalogue_multiple_files(self):
        # Create multiple catalog files
        self._remove_catalogues()
        catalog_path1 = os.path.join(self.path_to_simulation, "input", "cat1.xml")
        catalog_path2 = os.path.join(self.path_to_simulation, "input", "cat2.xml")
        for p in (catalog_path1, catalog_path2):
            with open(p, "w") as fh:
                fh.write("<event>fake</event>")

        catalog = self.output._find_catalogue()
        # Current implementation returns None when multiple catalogues are found.
        self.assertIsNone(catalog)
        self._remove_catalogues()

    def test_find_outputs_basic_structure(self):
        """_find_outputs() always returns a dict with 'elements' and 'stations'."""
        outputs = self.output._find_outputs()

        self.assertIsInstance(outputs, dict)
        self.assertIn('elements', outputs)
        self.assertIn('stations', outputs)
        self.assertIsInstance(outputs['elements'], dict)
        self.assertIsInstance(outputs['stations'], dict)
        # The committed fixture has no output/ directory, so both should be empty.
        self.assertDictEqual(outputs['elements'], {})
        self.assertDictEqual(outputs['stations'], {})

    def _remove_catalogues(self):
        for f in glob.glob(os.path.join(self.path_to_simulation, "input", "*cat*.xml")):
            os.remove(f)
