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
from nomad.units import ureg
from fhiaimsparser.fhiaims_parser import FHIAimsParser


def approx(value, abs=0, rel=1e-6):
    return pytest.approx(value, abs=abs, rel=rel)


@pytest.fixture(scope='module')
def parser():
    return FHIAimsParser()


@pytest.fixture(scope='module')
def silicon(parser):
    archive = EntryArchive()
    parser.parse('tests/data/Si_band_dos/aims_CC.out', archive, None)
    return archive


def test_scf_spinpol(parser):
    archive = EntryArchive()
    parser.parse('tests/data/Fe_scf_spinpol/out.out', archive, None)

    assert archive.section_run[0].program_version == '151211'
    assert archive.section_run[0].time_run_wall_start.magnitude == approx(2.23485023e+08)

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
    assert sec_atom_type.atom_type_mass.magnitude == approx(9.27328042e-26)
    assert sec_atom_type.x_fhi_aims_section_controlInOut_atom_species[0].x_fhi_aims_controlInOut_species_cut_pot.magnitude == approx(3.5e-10)

    assert len(archive.section_run[0].section_system) == 1
    sec_system = archive.section_run[0].section_system[0]
    assert sec_system.lattice_vectors[1][1].magnitude == approx(-1.4e-10)
    assert sec_system.configuration_periodic_dimensions == [True, True, True]
    assert sec_system.atom_labels == ['Fe']

    assert len(archive.section_run[0].section_single_configuration_calculation) == 1
    sec_scc = archive.section_run[0].section_single_configuration_calculation[0]
    assert sec_scc.energy_XC_functional.magnitude == approx(-2.3433685e-16)
    assert sec_scc.single_configuration_calculation_converged
    assert sec_scc.energy_reference_fermi[1].magnitude == approx(-1.50351795e-18)
    sec_scfs = sec_scc.section_scf_iteration
    assert len(sec_scfs) == 15
    assert sec_scfs[12].energy_total_T0_scf_iteration.magnitude == approx(-5.56048676e-15)
    assert sec_scfs[5].energy_reference_lowest_unoccupied_iteration[0].magnitude == approx(-1.42557688e-18)
    assert sec_scfs[7].energy_change_scf_iteration.magnitude == approx(9.43361602e-22)
    assert len(sec_scfs[14].section_eigenvalues_scf_iteration) == 1
    sec_eig = sec_scc.eigenvalues[0]
    assert np.shape(sec_eig.kpoints) == (4, 3)
    assert np.shape(sec_eig.band_energies[7].occupations) == (19,)
    assert sec_eig.band_energies[6].value[4].magnitude == approx(-1.1221523e-16)
    assert sec_eig.band_energies[3].occupations[9] == 1.0


def test_geomopt(parser):
    archive = EntryArchive()
    parser.parse('tests/data/Si_geomopt/out.out', archive, None)

    sec_methods = archive.section_run[0].section_method
    # one more since 0 is core settings
    assert len(sec_methods) == 7

    sec_sccs = archive.section_run[0].section_single_configuration_calculation
    assert len(sec_sccs) == 6

    assert np.shape(sec_sccs[1].eigenvalues[0].band_energies[0].value) == (20,)
    assert sec_sccs[2].energy_C.magnitude == approx(-9.34966824e-18)
    assert len(sec_sccs[3].section_scf_iteration) == 6
    assert np.max(sec_sccs[3].atom_forces_free_raw.magnitude) == approx(2.4933233e-11)
    assert np.max(sec_sccs[4].atom_forces_free.magnitude) == 0.


def test_band_spinpol(parser):
    archive = EntryArchive()
    parser.parse('tests/data/Fe_band_spinpol/Fe_band_structure_dos_spin.out', archive, None)

    assert len(archive.section_run[0].section_single_configuration_calculation) == 1
    sec_scc = archive.section_run[0].section_single_configuration_calculation[0]

    sec_k_band = sec_scc.band_structure_electronic[0]
    assert len(sec_k_band.band_structure_segment) == 3
    assert np.shape(sec_k_band.band_structure_segment[0].band_energies[29].value) == (19,)
    assert np.shape(sec_k_band.band_structure_segment[1].kpoints) == (15, 3)
    assert np.shape(sec_k_band.band_structure_segment[2].band_energies[29].occupations) == (19,)
    assert sec_k_band.band_structure_segment[0].band_energies[24].occupations[2] == approx(1.0)
    assert sec_k_band.band_structure_segment[1].band_energies[3].value[5].magnitude == approx(-1.54722007e-17)
    assert sec_k_band.band_structure_segment[2].kpoints[14][2] == approx(0.5)

    sec_dos = sec_scc.dos_electronic[0]
    assert np.shape(sec_dos.energies) == (50,)
    assert np.shape(sec_dos.total[1].value) == (50,)
    assert sec_dos.energies[46].magnitude == approx(-1.1999976e-18)
    assert sec_dos.total[0].value[46].magnitude == approx(1.2418253e+19)
    assert sec_dos.total[1].value[15].magnitude == approx(3.91517047e+19)


def test_band_silicon(silicon):
    """Tests that the band structure of silicon is parsed correctly.
    """
    scc = silicon.section_run[-1].section_single_configuration_calculation[0]
    band = scc.band_structure_electronic[-1]
    segments = band.band_structure_segment
    energies = []
    for segment in segments:
        energies.append([b.value.to(ureg.electron_volt).magnitude for b in segment.band_energies])
    energies = np.array(energies)

    # Check that an energy reference is reported
    energy_reference = scc.energy_reference_fermi
    assert energy_reference is not None
    energy_reference = energy_reference.to(ureg.electron_volt).magnitude

    # Check that an approporiately sized band gap is found at the given
    # reference energy
    energies = energies.flatten()
    energies.sort()
    lowest_unoccupied_index = np.searchsorted(energies, energy_reference, "right")[0]
    highest_occupied_index = lowest_unoccupied_index - 1
    gap = energies[lowest_unoccupied_index] - energies[highest_occupied_index]
    assert gap == approx(0.60684)


def test_dos_silicon(silicon):
    """Tests that the DOS of silicon is parsed correctly.
    """
    scc = silicon.section_run[-1].section_single_configuration_calculation[0]
    dos = scc.dos_electronic[-1]
    energies = dos.energies.to(ureg.electron_volt).magnitude
    values = np.array([d.value for d in dos.total])

    # Check that an energy reference is reported
    energy_reference = scc.energy_reference_fermi
    assert energy_reference is not None
    energy_reference = energy_reference.to(ureg.electron_volt).magnitude

    # Check that an approporiately sized band gap is found at the given
    # reference energy
    nonzero = np.unique(values.nonzero())
    energies = energies[nonzero]
    energies.sort()
    lowest_unoccupied_index = np.searchsorted(energies, energy_reference, "right")[0]
    highest_occupied_index = lowest_unoccupied_index - 1
    gap = energies[lowest_unoccupied_index] - energies[highest_occupied_index]
    assert gap == approx(0.54054054)


def test_dos(parser):
    archive = EntryArchive()
    parser.parse('tests/data/ClNa_dos/ClNa_dos.out', archive, None)

    sec_scc = archive.section_run[0].section_single_configuration_calculation[0]
    sec_dos = sec_scc.dos_electronic[0]
    assert np.shape(sec_dos.energies) == (50,)
    assert np.shape(sec_dos.total[0].value) == (50,)

    sec_species_dos = sec_dos.species_projected
    assert np.shape(sec_species_dos[7].value) == (50,)
    assert sec_species_dos[0].value[44].magnitude == approx(3.89674869e+18)
    assert sec_species_dos[1].value[37].magnitude == approx(7.85534487e+17)
    assert sec_species_dos[4].value[3].magnitude == approx(4.49311696e+17)
    assert sec_species_dos[7].value[5].magnitude == approx(2.46401047e+16)


def test_md(parser):
    archive = EntryArchive()
    parser.parse('tests/data/HO_md/H2O_periodic_MD.out', archive, None)

    sec_sccs = archive.section_run[0].section_single_configuration_calculation
    assert len(sec_sccs) == 6
    assert len(sec_sccs[1].section_scf_iteration) == 14
    assert sec_sccs[4].energy_sum_eigenvalues.magnitude == approx(-1.82257271e-16)

    sec_systems = archive.section_run[0].section_system
    assert len(sec_systems) == 6
    assert sec_systems[2].atom_velocities[1][2].magnitude == approx(-1195.746888)
    assert sec_systems[4].atom_positions[2][0].magnitude == approx(2.00130424e-10)


def test_gw(parser):
    archive = EntryArchive()
    parser.parse('tests/data/He_gw/He_scGW_ontop_PBE.out', archive, None)

    sec_methods = archive.section_run[0].section_method
    assert len(sec_methods) == 2
    assert sec_methods[1].electronic_structure_method == 'scGW'

    sec_scfs = archive.section_run[0].section_single_configuration_calculation[0].section_scf_iteration
    assert len(sec_scfs) == 6
    assert sec_scfs[1].x_fhi_aims_scgw_galitskii_migdal_total_energy.magnitude == approx(-1.28528018e-17)
    assert sec_scfs[4].x_fhi_aims_single_particle_energy.magnitude == approx(-4.96262869e-18)
