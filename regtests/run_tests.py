"""
This is a module for unit testing the BigDFT parser. The unit tests are run with
a custom backend that outputs the results directly into native python object for
easier and faster analysis.

Each property that has an enumerable list of different possible options is
assigned a new test class, that should ideally test through all the options.

The properties that can have non-enumerable values will be tested only for one
specific case inside a test class that is designed for a certain type of run
(MD, optimization, QM/MM, etc.)
"""
import os
import unittest
import logging
import numpy as np
from fhiaimsparser import FHIaimsParser


#===============================================================================
def get_results(rel_path):
    """Get the given result from the calculation in the given folder by using
    the Analyzer in the nomadtoolkit package. Tries to optimize the parsing by
    giving the metainfo_to_keep argument.

    Args:
        rel_path: The path to the output file. Relative to the directory of
            this script.
        metaname: The quantity to extract.
    """
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, rel_path)
    parser = FHIaimsParser(filename, None, debug=False, log_level=logging.WARNING)
    results = parser.parse()
    return results


#===============================================================================
def get_result(folder, metaname, optimize=True):
    if optimize:
        results = get_results(folder, None)
    else:
        results = get_results(folder)
    result = results[metaname]
    return result


#===============================================================================
class TestCrystal(unittest.TestCase):
    """Tests that the parser can handle single point calculations.
    """
    @classmethod
    def setUpClass(cls):
        cls.results = get_results("crystal/Si_band_structure.out")

    def test_program_name(self):
        result = self.results["program_name"]
        self.assertEqual(result, "FHI-aims")

    def test_program_basis_set(self):
        result = self.results["program_basis_set_type"]
        self.assertEqual(result, "numeric AOs")

    def test_electronic_structure_method(self):
        result = self.results["electronic_structure_method"]
        self.assertEqual(result, "DFT")

    def test_configuration_periodic_dimensions(self):
        result = self.results["configuration_periodic_dimensions"]
        self.assertTrue(np.array_equal(result, np.array([True, True, True])))


#===============================================================================
class TestMolecule(unittest.TestCase):
    """Tests that the parser can handle molecular systems.
    """
    @classmethod
    def setUpClass(cls):
        cls.results = get_results("molecule/output.dat")

    def test_program_name(self):
        result = self.results["program_name"]
        self.assertEqual(result, "FHI-aims")

    def test_program_basis_set(self):
        result = self.results["program_basis_set_type"]
        self.assertEqual(result, "numeric AOs")

    def test_electronic_structure_method(self):
        result = self.results["electronic_structure_method"]
        self.assertEqual(result, "DFT")

    def test_configuration_periodic_dimensions(self):
        result = self.results["configuration_periodic_dimensions"]
        self.assertTrue(np.array_equal(result, np.array([False, False, False])))

#===============================================================================
if __name__ == '__main__':
    suites = []
    suites.append(unittest.TestLoader().loadTestsFromTestCase(TestCrystal))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(TestMolecule))

    alltests = unittest.TestSuite(suites)
    unittest.TextTestRunner(verbosity=0).run(alltests)
