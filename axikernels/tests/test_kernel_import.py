import importlib
import importlib.util
import sys
import unittest


class TestKernelImport(unittest.TestCase):
    def test_import_kernel_without_visualization_dependencies(self):
        sys.modules.pop("axikernels.core.kernels", None)
        sys.modules.pop("axikernels.core.kernels.kernel", None)
        sys.modules.pop("axikernels.core.kernels.objective_function", None)

        module = importlib.import_module("axikernels.core.kernels")

        self.assertTrue(hasattr(module, "Kernel"))

        if importlib.util.find_spec("ruamel") is None:
            self.assertIsNone(module.L2ObjectiveFunction)
        else:
            self.assertIsNotNone(module.L2ObjectiveFunction)