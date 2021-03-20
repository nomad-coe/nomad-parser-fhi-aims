#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD. See https://nomad-lab.eu for further info.
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

import pytest
import numpy as np

from nomad.datamodel import EntryArchive
from fhiaimsparser.fhiaims_parser import FHIAimsParser


@pytest.fixture(scope='module')
def parser():
    return FHIAimsParser()


def test_scf_spinpol(parser):
    archive = EntryArchive()
    parser.parse('tests/data/Fe_scf_spinpol/out.out', archive, None)

    assert archive.section_run[0].program_version == '151211'
    assert pytest.approx(archive.section_run[0].time_run_wall_start.magnitude, 2.23485023e+08)

    assert len(archive.section_run[0].section_method) == 2
    sec_method = archive.section_run[0].section_method[0]
    assert sec_method.number_of_spin_channels == 2
    assert sec_method.relativity_method == 'scalar_relativistic_atomic_ZORA'
    assert sec_method.section_XC_functionals[0].XC_functional_name == 'LDA_C_PW'
    sec_basis_func = sec_method.x_fhi_aims_section_controlIn_basis_set[0].x_fhi_aims_section_controlIn_basis_func
    assert len(sec_basis_func) == 10
    assert sec_basis_func[2].x_fhi_aims_controlIn_basis_func_radius == 6.0
    assert sec_basis_func[6].x_fhi_aims_controlIn_basis_func_type == 'hydro'
    sec_atom_type = archive.section_run[0].section_topology[0].section_atom_type[0]
    assert pytest.approx(sec_atom_type.atom_type_mass.magnitude, 9.27328042e-26)
    assert pytest.approx(sec_atom_type.x_fhi_aims_section_controlInOut_atom_species[0].x_fhi_aims_controlInOut_species_cut_pot.magnitude, 3.5e-10)

    assert len(archive.section_run[0].section_system) == 1
    sec_system = archive.section_run[0].section_system[0]
    assert pytest.approx(sec_system.lattice_vectors[1][1].magnitude, 1.4e-10)
    assert sec_system.configuration_periodic_dimensions == [True, True, True]
    assert sec_system.atom_labels == ['Fe']

    assert len(archive.section_run[0].section_single_configuration_calculation) == 1
    sec_scc = archive.section_run[0].section_single_configuration_calculation[0]
    assert pytest.approx(sec_scc.energy_XC_functional.magnitude, -2.3433685e-16)
    assert sec_scc.single_configuration_calculation_converged
    assert pytest.approx(sec_scc.energy_reference_fermi[1].magnitude, -1.50351795e-18)
    sec_scfs = sec_scc.section_scf_iteration
    assert len(sec_scfs) == 15
    assert pytest.approx(sec_scfs[12].energy_total_T0_scf_iteration.magnitude, -5.56048676e-15)
    assert pytest.approx(sec_scfs[5].energy_reference_lowest_unoccupied_iteration[0].magnitude, -1.42557688e-18)
    assert pytest.approx(sec_scfs[7].energy_change_scf_iteration.magnitude, -1.42557688e-18)
    assert len(sec_scfs[14].section_eigenvalues_scf_iteration) == 1
    sec_eig = sec_scc.section_eigenvalues[0]
    assert np.shape(sec_eig.eigenvalues_kpoints) == (4, 3)
    assert np.shape(sec_eig.eigenvalues_occupation) == (2, 4, 19)
    assert pytest.approx(sec_eig.eigenvalues_values[1][2][4].magnitude, -1.1221523e-16)
    assert sec_eig.eigenvalues_occupation[0][3][9] == 1.0


def test_geomopt(parser):
    archive = EntryArchive()
    parser.parse('tests/data/Si_geomopt/out.out', archive, None)

    sec_methods = archive.section_run[0].section_method
    # one more since 0 is core settings
    assert len(sec_methods) == 7

    sec_sccs = archive.section_run[0].section_single_configuration_calculation
    assert len(sec_sccs) == 6

    assert np.shape(sec_sccs[1].section_eigenvalues[0].eigenvalues_values) == (1, 1, 20)
    assert pytest.approx(sec_sccs[2].energy_C.magnitude, -9.35075519e-18)
    assert len(sec_sccs[3].section_scf_iteration) == 6
    assert pytest.approx(np.max(sec_sccs[3].atom_forces_free_raw.magnitude), 1.02243543e-11)
    assert np.max(sec_sccs[4].atom_forces_free.magnitude) == 0.


def test_band_spinpol(parser):
    archive = EntryArchive()
    parser.parse('tests/data/Fe_band_spinpol/Fe_band_structure_dos_spin.out', archive, None)

    assert len(archive.section_run[0].section_single_configuration_calculation) == 1
    sec_scc = archive.section_run[0].section_single_configuration_calculation[0]

    sec_k_band = sec_scc.section_k_band[0]
    assert len(sec_k_band.section_k_band_segment) == 3
    assert np.shape(sec_k_band.section_k_band_segment[0].band_energies) == (2, 15, 19)
    assert np.shape(sec_k_band.section_k_band_segment[1].band_k_points) == (15, 3)
    assert np.shape(sec_k_band.section_k_band_segment[2].band_occupations) == (2, 15, 19)
    assert pytest.approx(sec_k_band.section_k_band_segment[0].band_occupations[1][9][2], 1.0)
    assert pytest.approx(sec_k_band.section_k_band_segment[1].band_energies[0][3][5].magnitude, -1.39686827e-17)
    assert pytest.approx(sec_k_band.section_k_band_segment[2].band_k_points[14][2], 0.5)

    sec_dos = sec_scc.section_dos[0]
    assert np.shape(sec_dos.dos_energies) == (50,)
    assert np.shape(sec_dos.dos_values) == (2, 50)
    assert pytest.approx(sec_dos.dos_energies[46].magnitude, -2.06321113e-18)
    assert pytest.approx(sec_dos.dos_values[0][46], 1.2418252951253564e-11)
    assert pytest.approx(sec_dos.dos_values[1][15], 3.9151704709731774e-11)

    sec_atom_dos = sec_scc.section_atom_projected_dos[0]
    assert np.shape(sec_atom_dos.atom_projected_dos_energies) == (50,)
    assert np.shape(sec_atom_dos.atom_projected_dos_values_lm) == (4, 2, 1, 50)
    assert np.shape(sec_atom_dos.atom_projected_dos_values_total) == (2, 1, 50)
    assert pytest.approx(sec_atom_dos.atom_projected_dos_energies[49].magnitude, -1.12152364e-18)
    assert pytest.approx(sec_atom_dos.atom_projected_dos_values_lm[2][1][0][22], 6.61986474e+18)
    assert pytest.approx(sec_atom_dos.atom_projected_dos_values_lm[0][0][0][8], 3.35029976e+17)
    assert pytest.approx(sec_atom_dos.atom_projected_dos_values_total[0][0][40], 9.950877863070872e+17)


def test_dos(parser):
    archive = EntryArchive()
    parser.parse('tests/data/ClNa_dos/ClNa_dos.out', archive, None)

    sec_scc = archive.section_run[0].section_single_configuration_calculation[0]
    sec_dos = sec_scc.section_dos[0]
    assert np.shape(sec_dos.dos_energies) == (50,)
    assert np.shape(sec_dos.dos_values) == (1, 50)

    sec_species_dos = sec_scc.section_species_projected_dos[0]
    assert np.shape(sec_species_dos.species_projected_dos_energies) == (50,)
    assert np.shape(sec_species_dos.species_projected_dos_values_lm) == (3, 1, 2, 50)
    assert np.shape(sec_species_dos.species_projected_dos_values_total) == (1, 2, 50)
    assert pytest.approx(sec_species_dos.species_projected_dos_energies[9].magnitude, -1.35040602e-18)
    assert pytest.approx(sec_species_dos.species_projected_dos_values_total[0][0][44], 3.89674869e+18)
    assert pytest.approx(sec_species_dos.species_projected_dos_values_total[0][1][37], 7.85534487e+17)
    assert pytest.approx(sec_species_dos.species_projected_dos_values_lm[1][0][0][3], 4.49311696e+17)
    assert pytest.approx(sec_species_dos.species_projected_dos_values_lm[2][0][1][5], 2.46401047e+16)


def test_md(parser):
    archive = EntryArchive()
    parser.parse('tests/data/HO_md/H2O_periodic_MD.out', archive, None)

    sec_sccs = archive.section_run[0].section_single_configuration_calculation
    assert len(sec_sccs) == 6
    assert len(sec_sccs[1].section_scf_iteration) == 14
    assert pytest.approx(sec_sccs[4].energy_sum_eigenvalues.magnitude, -1.82257271e-16)

    sec_systems = archive.section_run[0].section_system
    assert len(sec_systems) == 6
    assert pytest.approx(sec_systems[2].atom_velocities[1][2].magnitude, -1195.746888)
    assert pytest.approx(sec_systems[4].atom_positions[2][0].magnitude, 2.00130424e-10)


def test_gw(parser):
    archive = EntryArchive()
    parser.parse('tests/data/He_gw/He_scGW_ontop_PBE.out', archive, None)

    sec_methods = archive.section_run[0].section_method
    assert len(sec_methods) == 2
    assert sec_methods[1].electronic_structure_method == 'scGW'

    sec_scfs = archive.section_run[0].section_single_configuration_calculation[0].section_scf_iteration
    assert len(sec_scfs) == 6
    assert pytest.approx(sec_scfs[1].x_fhi_aims_scgw_galitskii_migdal_total_energy.magnitude, -1.28528018e-17)
    assert pytest.approx(sec_scfs[4].x_fhi_aims_single_particle_energy.magnitude, -4.96262869e-18)
