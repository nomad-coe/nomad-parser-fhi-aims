#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD.
# See https://nomad-lab.eu for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
This is a module that defines unit tests for the parser. The unit tests are run
with a custom backend that outputs the results directly into native python
object for easier and faster testing.
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
class TestG0W0(unittest.TestCase):
    """Tests that the parser can handle G0W0 calculations.
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
        self.assertEqual(result, "G0W0")

    def test_configuration_periodic_dimensions(self):
        result = self.results["configuration_periodic_dimensions"]
        self.assertTrue(np.array_equal(result, np.array([False, False, False])))


#===============================================================================
if __name__ == '__main__':
    suites = []
    suites.append(unittest.TestLoader().loadTestsFromTestCase(TestCrystal))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(TestMolecule))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(TestG0W0))

    alltests = unittest.TestSuite(suites)
    unittest.TextTestRunner(verbosity=0).run(alltests)
