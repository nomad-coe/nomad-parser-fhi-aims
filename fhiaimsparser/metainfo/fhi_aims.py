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
import numpy as np            # pylint: disable=unused-import
import typing                 # pylint: disable=unused-import
from nomad.metainfo import (  # pylint: disable=unused-import
    MSection, MCategory, Category, Package, Quantity, Section, SubSection, SectionProxy,
    Reference
)
from nomad.units import ureg
from nomad.metainfo.legacy import LegacyDefinition

from nomad.datamodel.metainfo import public
from nomad.datamodel.metainfo import common

m_package = Package(
    name='fhi_aims_nomadmetainfo_json',
    description='None',
    a_legacy=LegacyDefinition(name='fhi_aims.nomadmetainfo.json'))


class x_fhi_aims_controlIn_method(MCategory):
    '''
    Parameters of control.in belonging to section method.
    '''

    m_def = Category(
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlIn_method'))


class x_fhi_aims_controlInOut_method(MCategory):
    '''
    Parameters of aims output of parsed control.in belonging to section method.
    '''

    m_def = Category(
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlInOut_method'))


class x_fhi_aims_controlIn_run(MCategory):
    '''
    Parameters of control.in belonging to settings run.
    '''

    m_def = Category(
        categories=[public.settings_run],
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlIn_run'))


class x_fhi_aims_controlInOut_run(MCategory):
    '''
    Parameters of aims output of parsed control.in belonging to settings run.
    '''

    m_def = Category(
        categories=[public.settings_run],
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlInOut_run'))


class x_fhi_aims_section_controlIn_basis_func(MSection):
    '''
    definition of a single basis function in the basis set
    '''

    m_def = Section(validate=False, a_legacy=LegacyDefinition(name='x_fhi_aims_section_controlIn_basis_func'))

    x_fhi_aims_controlIn_basis_func_l = Quantity(
        type=str,
        shape=[],
        description='''
        -
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlIn_basis_func_l'))

    x_fhi_aims_controlIn_basis_func_n = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        -
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlIn_basis_func_n'))

    x_fhi_aims_controlIn_basis_func_radius = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        -
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlIn_basis_func_radius'))

    x_fhi_aims_controlIn_basis_func_type = Quantity(
        type=str,
        shape=[],
        description='''
        -
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlIn_basis_func_type'))


class x_fhi_aims_section_controlIn_basis_set(MSection):
    '''
    -
    '''

    m_def = Section(validate=False, a_legacy=LegacyDefinition(name='x_fhi_aims_section_controlIn_basis_set'))

    x_fhi_aims_controlIn_angular_grids_method = Quantity(
        type=str,
        shape=[],
        description='''
        angular grids method (specifed or auto)
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlIn_angular_grids_method'))

    x_fhi_aims_controlIn_basis_dep_cutoff = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        cutoff for the dependent basis
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlIn_basis_dep_cutoff'))

    x_fhi_aims_controlIn_cut_pot = Quantity(
        type=np.dtype(np.float64),
        shape=[3],
        description='''
        cut\\_pot parameters
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlIn_cut_pot'))

    x_fhi_aims_controlIn_cut_pot1 = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        first parameter of cut\\_pot
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlIn_cut_pot1'))

    x_fhi_aims_controlIn_cut_pot2 = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        second parameter of cut\\_pot
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlIn_cut_pot2'))

    x_fhi_aims_controlIn_cut_pot3 = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        third parameter of cut\\_pot
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlIn_cut_pot3'))

    x_fhi_aims_controlIn_division1 = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        first parameter of division (position)
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlIn_division1'))

    x_fhi_aims_controlIn_division2 = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        second parameter of division (n points)
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlIn_division2'))

    x_fhi_aims_controlIn_division = Quantity(
        type=np.dtype(np.float64),
        shape=['x_fhi_aims_controlIn_number_of_basis_func', 2],
        description='''
        division parameters
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlIn_division'))

    x_fhi_aims_controlIn_number_of_basis_func = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        number of basis functions
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlIn_number_of_basis_func'))

    x_fhi_aims_controlIn_l_hartree = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        angular leven for the hartreee part
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlIn_l_hartree'))

    x_fhi_aims_controlIn_mass = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        mass of the nucleus in atomic mass units
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlIn_mass'))

    x_fhi_aims_controlIn_nucleus = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        charge of the nucleus
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlIn_nucleus'))

    x_fhi_aims_controlIn_outer_grid = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        outer grid
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlIn_outer_grid'))

    x_fhi_aims_controlIn_radial_base = Quantity(
        type=np.dtype(np.float64),
        shape=[2],
        description='''
        radial\\_base parameters
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlIn_radial_base'))

    x_fhi_aims_controlIn_radial_base1 = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        first parameter of radial\\_base
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlIn_radial_base1'))

    x_fhi_aims_controlIn_radial_base2 = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        second parameter of radial\\_base
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlIn_radial_base2'))

    x_fhi_aims_controlIn_radial_multiplier = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        radial multiplier
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlIn_radial_multiplier'))

    x_fhi_aims_controlIn_species_name = Quantity(
        type=str,
        shape=[],
        description='''
        -
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlIn_species_name'))

    x_fhi_aims_section_controlIn_basis_func = SubSection(
        sub_section=SectionProxy('x_fhi_aims_section_controlIn_basis_func'),
        repeats=True,
        a_legacy=LegacyDefinition(name='x_fhi_aims_section_controlIn_basis_func'))


class x_fhi_aims_section_controlInOut_atom_species(MSection):
    '''
    -
    '''

    m_def = Section(validate=False, a_legacy=LegacyDefinition(name='x_fhi_aims_section_controlInOut_atom_species'))

    x_fhi_aims_controlInOut_pure_gaussian = Quantity(
        type=str,
        shape=[],
        description='''
        -
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlInOut_pure_gaussian'))

    x_fhi_aims_controlInOut_species_charge = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='coulomb',
        description='''
        -
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlInOut_species_charge'))

    x_fhi_aims_controlInOut_species_cut_pot_scale = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        -
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlInOut_species_cut_pot_scale'))

    x_fhi_aims_controlInOut_species_cut_pot_width = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='meter',
        description='''
        -
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlInOut_species_cut_pot_width'))

    x_fhi_aims_controlInOut_species_cut_pot = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='meter',
        description='''
        -
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlInOut_species_cut_pot'))

    x_fhi_aims_controlInOut_species_mass = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='kilogram',
        description='''
        -
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlInOut_species_mass'))

    x_fhi_aims_controlInOut_species_name = Quantity(
        type=str,
        shape=[],
        description='''
        -
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlInOut_species_name'))

    x_fhi_aims_section_controlInOut_basis_func = SubSection(
        sub_section=SectionProxy('x_fhi_aims_section_controlInOut_basis_func'),
        repeats=True,
        a_legacy=LegacyDefinition(name='x_fhi_aims_section_controlInOut_basis_func'))

    x_fhi_aims_section_vdW_TS = SubSection(
        sub_section=SectionProxy('x_fhi_aims_section_vdW_TS'),
        repeats=True,
        a_legacy=LegacyDefinition(name='x_fhi_aims_section_vdW_TS'))


class x_fhi_aims_section_controlInOut_basis_func(MSection):
    '''
    -
    '''

    m_def = Section(validate=False, a_legacy=LegacyDefinition(name='x_fhi_aims_section_controlInOut_basis_func'))

    x_fhi_aims_controlInOut_basis_func_eff_charge = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        -
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlInOut_basis_func_eff_charge'))

    x_fhi_aims_controlInOut_basis_func_gauss_alpha = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='1 / meter ** 2',
        description='''
        -
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlInOut_basis_func_gauss_alpha'))

    x_fhi_aims_controlInOut_basis_func_gauss_l = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        -
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlInOut_basis_func_gauss_l'))

    x_fhi_aims_controlInOut_basis_func_gauss_N = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        -
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlInOut_basis_func_gauss_N'))

    x_fhi_aims_controlInOut_basis_func_gauss_weight = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        -
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlInOut_basis_func_gauss_weight'))

    x_fhi_aims_controlInOut_basis_func_l = Quantity(
        type=str,
        shape=[],
        description='''
        -
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlInOut_basis_func_l'))

    x_fhi_aims_controlInOut_basis_func_n = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        -
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlInOut_basis_func_n'))

    x_fhi_aims_controlInOut_basis_func_occ = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        -
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlInOut_basis_func_occ'))

    x_fhi_aims_controlInOut_basis_func_primitive_gauss_alpha = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='1 / meter ** 2',
        description='''
        -
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlInOut_basis_func_primitive_gauss_alpha'))

    x_fhi_aims_controlInOut_basis_func_radius = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        -
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlInOut_basis_func_radius'))

    x_fhi_aims_controlInOut_basis_func_type = Quantity(
        type=str,
        shape=[],
        description='''
        -
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlInOut_basis_func_type'))


class x_fhi_aims_section_eigenvalues_group_perturbativeGW(MSection):
    '''
    section for full list of eigenvalues for different spin and kpoints from a
    perturbative GW calculation
    '''

    m_def = Section(validate=False, a_legacy=LegacyDefinition(name='x_fhi_aims_section_eigenvalues_group_perturbativeGW'))

    x_fhi_aims_section_eigenvalues_spin_perturbativeGW = SubSection(
        sub_section=SectionProxy('x_fhi_aims_section_eigenvalues_spin_perturbativeGW'),
        repeats=True,
        a_legacy=LegacyDefinition(name='x_fhi_aims_section_eigenvalues_spin_perturbativeGW'))


class x_fhi_aims_section_eigenvalues_group_ZORA(MSection):
    '''
    section for full list of eigenvalues for different spin and kpoints of scaled ZORA
    '''

    m_def = Section(validate=False, a_legacy=LegacyDefinition(name='x_fhi_aims_section_eigenvalues_group_ZORA'))

    x_fhi_aims_section_eigenvalues_spin_ZORA = SubSection(
        sub_section=SectionProxy('x_fhi_aims_section_eigenvalues_spin_ZORA'),
        repeats=True,
        a_legacy=LegacyDefinition(name='x_fhi_aims_section_eigenvalues_spin_ZORA'))


class x_fhi_aims_section_eigenvalues_group(MSection):
    '''
    section for full list of eigenvalues for different spin and kpoints
    '''

    m_def = Section(validate=False, a_legacy=LegacyDefinition(name='x_fhi_aims_section_eigenvalues_group'))

    x_fhi_aims_section_eigenvalues_spin = SubSection(
        sub_section=SectionProxy('x_fhi_aims_section_eigenvalues_spin'),
        repeats=True,
        a_legacy=LegacyDefinition(name='x_fhi_aims_section_eigenvalues_spin'))


class x_fhi_aims_section_eigenvalues_list_perturbativeGW(MSection):
    '''
    section for one list of eigenvalues from a perturbative GW calculation
    '''

    m_def = Section(validate=False, a_legacy=LegacyDefinition(name='x_fhi_aims_section_eigenvalues_list_perturbativeGW'))

    x_fhi_aims_eigenvalue_correlation_perturbativeGW = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='joule',
        description='''
        Correlation energy at a given eigenstate from perturbative GW
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_eigenvalue_correlation_perturbativeGW'))

    x_fhi_aims_eigenvalue_ExactExchange_perturbativeGW = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='joule',
        description='''
        Exact exchange energy at given eigenstate from perturbative GW
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_eigenvalue_ExactExchange_perturbativeGW'))

    x_fhi_aims_eigenvalue_ks_ExchangeCorrelation = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='joule',
        description='''
        KS exchange correlation energy at a given eigenstate needed to calculate the
        quasi-particle energy in perturbative GW
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_eigenvalue_ks_ExchangeCorrelation'))

    x_fhi_aims_eigenvalue_ks_GroundState = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='joule',
        description='''
        KS ground state energy at a given eigenstate needed in perturbative GW
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_eigenvalue_ks_GroundState'))

    x_fhi_aims_eigenvalue_occupation_perturbativeGW = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        Occupation of single eigenfunction of perturbative GW
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_eigenvalue_occupation_perturbativeGW'))

    x_fhi_aims_eigenvalue_quasiParticle_energy = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='joule',
        description='''
        Quasiparticle energy at a given eigenstate from perturbative GW
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_eigenvalue_quasiParticle_energy'))


class x_fhi_aims_section_eigenvalues_list_ZORA(MSection):
    '''
    section for one list of eigenvalues at specific kpoint and spin of scaled ZORA
    '''

    m_def = Section(validate=False, a_legacy=LegacyDefinition(name='x_fhi_aims_section_eigenvalues_list_ZORA'))

    x_fhi_aims_eigenvalue_eigenvalue_ZORA = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='joule',
        description='''
        Single eigenvalue of scaled ZORA
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_eigenvalue_eigenvalue_ZORA'))

    x_fhi_aims_eigenvalue_occupation_ZORA = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        Occupation of single eigenfunction of scaled ZORA
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_eigenvalue_occupation_ZORA'))


class x_fhi_aims_section_eigenvalues_list(MSection):
    '''
    section for one list of eigenvalues at specific kpoint and spin
    '''

    m_def = Section(validate=False, a_legacy=LegacyDefinition(name='x_fhi_aims_section_eigenvalues_list'))

    x_fhi_aims_eigenvalue_eigenvalue = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='joule',
        description='''
        Single eigenvalue
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_eigenvalue_eigenvalue'))

    x_fhi_aims_eigenvalue_occupation = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        Occupation of single eigenfunction
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_eigenvalue_occupation'))


class x_fhi_aims_section_eigenvalues_spin_perturbativeGW(MSection):
    '''
    section for one spin orientation from a perturbative GW calculation
    '''

    m_def = Section(validate=False, a_legacy=LegacyDefinition(name='x_fhi_aims_section_eigenvalues_spin_perturbativeGW'))


class x_fhi_aims_section_eigenvalues_spin_ZORA(MSection):
    '''
    section for one spin orientation of scaled ZORA
    '''

    m_def = Section(validate=False, a_legacy=LegacyDefinition(name='x_fhi_aims_section_eigenvalues_spin_ZORA'))

    x_fhi_aims_eigenvalue_kpoint1_ZORA = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        Component 1 of kpoints on which the eigenvalues were evaluated of scaled ZORA
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_eigenvalue_kpoint1_ZORA'))

    x_fhi_aims_eigenvalue_kpoint2_ZORA = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        Component 2 of kpoints on which the eigenvalues were evaluated of scaled ZORA
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_eigenvalue_kpoint2_ZORA'))

    x_fhi_aims_eigenvalue_kpoint3_ZORA = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        Component 3 of kpoints on which the eigenvalues were evaluated of scaled ZORA
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_eigenvalue_kpoint3_ZORA'))

    x_fhi_aims_section_eigenvalues_list_ZORA = SubSection(
        sub_section=SectionProxy('x_fhi_aims_section_eigenvalues_list_ZORA'),
        repeats=True,
        a_legacy=LegacyDefinition(name='x_fhi_aims_section_eigenvalues_list_ZORA'))


class x_fhi_aims_section_eigenvalues_spin(MSection):
    '''
    section for one spin orientation
    '''

    m_def = Section(validate=False, a_legacy=LegacyDefinition(name='x_fhi_aims_section_eigenvalues_spin'))

    x_fhi_aims_eigenvalue_kpoint1 = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        Component 1 of kpoints on which the eigenvalues were evaluated
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_eigenvalue_kpoint1'))

    x_fhi_aims_eigenvalue_kpoint2 = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        Component 2 of kpoints on which the eigenvalues were evaluated
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_eigenvalue_kpoint2'))

    x_fhi_aims_eigenvalue_kpoint3 = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        Component 3 of kpoints on which the eigenvalues were evaluated
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_eigenvalue_kpoint3'))

    x_fhi_aims_section_eigenvalues_list_perturbativeGW = SubSection(
        sub_section=SectionProxy('x_fhi_aims_section_eigenvalues_list_perturbativeGW'),
        repeats=True,
        a_legacy=LegacyDefinition(name='x_fhi_aims_section_eigenvalues_list_perturbativeGW'))

    x_fhi_aims_section_eigenvalues_list = SubSection(
        sub_section=SectionProxy('x_fhi_aims_section_eigenvalues_list'),
        repeats=True,
        a_legacy=LegacyDefinition(name='x_fhi_aims_section_eigenvalues_list'))


class x_fhi_aims_section_eigenvalues_ZORA(MSection):
    '''
    section for gathering eigenvalues of scaled ZORA
    '''

    m_def = Section(validate=False, a_legacy=LegacyDefinition(name='x_fhi_aims_section_eigenvalues_ZORA'))

    x_fhi_aims_section_eigenvalues_group_ZORA = SubSection(
        sub_section=SectionProxy('x_fhi_aims_section_eigenvalues_group_ZORA'),
        repeats=False,
        a_legacy=LegacyDefinition(name='x_fhi_aims_section_eigenvalues_group_ZORA'))


class x_fhi_aims_section_MD_detect(MSection):
    '''
    Section to detect MD immediately during parsing of controlInOut
    '''

    m_def = Section(validate=False, a_legacy=LegacyDefinition(name='x_fhi_aims_section_MD_detect'))


class x_fhi_aims_section_parallel_task_assignement(MSection):
    '''
    -
    '''

    m_def = Section(validate=False, a_legacy=LegacyDefinition(name='x_fhi_aims_section_parallel_task_assignement'))

    x_fhi_aims_parallel_task_host = Quantity(
        type=str,
        shape=[],
        description='''
        -
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_parallel_task_host'))

    x_fhi_aims_parallel_task_nr = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        -
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_parallel_task_nr'))


class x_fhi_aims_section_parallel_tasks(MSection):
    '''
    -
    '''

    m_def = Section(validate=False, a_legacy=LegacyDefinition(name='x_fhi_aims_section_parallel_tasks'))

    x_fhi_aims_section_parallel_task_assignement = SubSection(
        sub_section=SectionProxy('x_fhi_aims_section_parallel_task_assignement'),
        repeats=True,
        a_legacy=LegacyDefinition(name='x_fhi_aims_section_parallel_task_assignement'))


class x_fhi_aims_section_vdW_TS(MSection):
    '''
    -
    '''

    m_def = Section(validate=False, a_legacy=LegacyDefinition(name='x_fhi_aims_section_vdW_TS'))

    x_fhi_aims_atom_type_vdW = Quantity(
        type=str,
        shape=[],
        description='''
        -
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_atom_type_vdW'))

    x_fhi_aims_free_atom_volume = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        -
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_free_atom_volume'))

    x_fhi_aims_hirschfeld_charge = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        -
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_hirschfeld_charge'))

    x_fhi_aims_hirschfeld_volume = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        -
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_hirschfeld_volume'))

    x_fhi_aims_vdW_energy_corr_TS = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='joule',
        description='''
        -
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_vdW_energy_corr_TS'))


class section_single_configuration_calculation(public.section_single_configuration_calculation):

    m_def = Section(validate=False, extends_base_section=True, a_legacy=LegacyDefinition(name='section_single_configuration_calculation'))

    x_fhi_aims_atom_forces_free_x = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='newton',
        description='''
        -
        ''',
        categories=[public.atom_forces_type],
        a_legacy=LegacyDefinition(name='x_fhi_aims_atom_forces_free_x'))

    x_fhi_aims_atom_forces_free_y = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='newton',
        description='''
        -
        ''',
        categories=[public.atom_forces_type],
        a_legacy=LegacyDefinition(name='x_fhi_aims_atom_forces_free_y'))

    x_fhi_aims_atom_forces_free_z = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='newton',
        description='''
        -
        ''',
        categories=[public.atom_forces_type],
        a_legacy=LegacyDefinition(name='x_fhi_aims_atom_forces_free_z'))

    x_fhi_aims_energy_C_LDA = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='joule',
        description='''
        Component of the correlation (C) energy at the LDA level calculated with the self
        consistent density of the target functional.
        ''',
        categories=[public.energy_value, public.energy_type_C, public.energy_component],
        a_legacy=LegacyDefinition(name='x_fhi_aims_energy_C_LDA'))

    x_fhi_aims_energy_X_LDA = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='joule',
        description='''
        Component of the exchange (X) energy at the LDA level calculated with the self
        consistent density of the target functional.
        ''',
        categories=[public.energy_value, public.energy_type_X, public.energy_component],
        a_legacy=LegacyDefinition(name='x_fhi_aims_energy_X_LDA'))

    x_fhi_aims_cube_filename = Quantity(
        type=str,
        shape=[],
        description='''
        filename of cube file
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_cube_filename'))

    x_fhi_aims_section_eigenvalues_group_perturbativeGW = SubSection(
        sub_section=SectionProxy('x_fhi_aims_section_eigenvalues_group_perturbativeGW'),
        repeats=False,
        a_legacy=LegacyDefinition(name='x_fhi_aims_section_eigenvalues_group_perturbativeGW'))

    x_fhi_aims_section_eigenvalues_ZORA = SubSection(
        sub_section=SectionProxy('x_fhi_aims_section_eigenvalues_ZORA'),
        repeats=False,
        a_legacy=LegacyDefinition(name='x_fhi_aims_section_eigenvalues_ZORA'))


class section_scf_iteration(public.section_scf_iteration):

    m_def = Section(validate=False, extends_base_section=True, a_legacy=LegacyDefinition(name='section_scf_iteration'))

    x_fhi_aims_atom_forces_raw_x = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='newton',
        description='''
        -
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_atom_forces_raw_x'))

    x_fhi_aims_atom_forces_raw_y = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='newton',
        description='''
        -
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_atom_forces_raw_y'))

    x_fhi_aims_atom_forces_raw_z = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='newton',
        description='''
        -
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_atom_forces_raw_z'))

    x_fhi_aims_energy_electrostatic_free_atom_scf_iteration = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='joule',
        description='''
        Electrostatic energy contributions from superposition of free atom densities
        during the scf iterations
        ''',
        categories=[public.energy_value, public.energy_component],
        a_legacy=LegacyDefinition(name='x_fhi_aims_energy_electrostatic_free_atom_scf_iteration'))

    x_fhi_aims_energy_scgw_correlation_energy = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='joule',
        description='''
        scGW correlation energy at each iteration
        ''',
        categories=[public.energy_value, public.energy_component],
        a_legacy=LegacyDefinition(name='x_fhi_aims_energy_scgw_correlation_energy'))

    x_fhi_aims_poles_fit_accuracy = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        Fit acccuracy for the Fast-Fourier Transforms necessary in the scGW formalism
        ''',
        categories=[public.energy_value, public.energy_component],
        a_legacy=LegacyDefinition(name='x_fhi_aims_poles_fit_accuracy'))

    x_fhi_aims_scf_date_start = Quantity(
        type=str,
        shape=[],
        description='''
        -
        ''',
        categories=[public.time_info, public.accessory_info],
        a_legacy=LegacyDefinition(name='x_fhi_aims_scf_date_start'))

    x_fhi_aims_scf_time_start = Quantity(
        type=str,
        shape=[],
        description='''
        -
        ''',
        categories=[public.time_info, public.accessory_info],
        a_legacy=LegacyDefinition(name='x_fhi_aims_scf_time_start'))

    x_fhi_aims_scgw_galitskii_migdal_total_energy = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='joule',
        description='''
        scGW total energy at each iteration calculated using the Galitskii-Migdal formula
        ''',
        categories=[public.energy_value, public.energy_component],
        a_legacy=LegacyDefinition(name='x_fhi_aims_scgw_galitskii_migdal_total_energy'))

    x_fhi_aims_scgw_hartree_energy_sum_eigenvalues_scf_iteration = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='joule',
        description='''
        scGW sum of eigenvalues calculated from the trace over the Hamiltonian times the
        Greens function matrices
        ''',
        categories=[public.energy_value, public.energy_component],
        a_legacy=LegacyDefinition(name='x_fhi_aims_scgw_hartree_energy_sum_eigenvalues_scf_iteration'))

    x_fhi_aims_scgw_kinetic_energy = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='joule',
        description='''
        scGW kinetic energy at each iteration
        ''',
        categories=[public.energy_value, public.energy_component],
        a_legacy=LegacyDefinition(name='x_fhi_aims_scgw_kinetic_energy'))

    x_fhi_aims_scgw_rpa_correlation_energy = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='joule',
        description='''
        The RPA correlation energy calculated from the Green's functions of the scGW at
        each iteration
        ''',
        categories=[public.energy_value, public.energy_component],
        a_legacy=LegacyDefinition(name='x_fhi_aims_scgw_rpa_correlation_energy'))

    x_fhi_aims_single_configuration_calculation_converged = Quantity(
        type=str,
        shape=[],
        description='''
        Determines whether a single configuration calculation is converged.
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_single_configuration_calculation_converged'))

    x_fhi_aims_single_particle_energy = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='joule',
        description='''
        scGW single particle energy at each iteration
        ''',
        categories=[public.energy_value, public.energy_component],
        a_legacy=LegacyDefinition(name='x_fhi_aims_single_particle_energy'))

    x_fhi_aims_section_eigenvalues_group = SubSection(
        sub_section=SectionProxy('x_fhi_aims_section_eigenvalues_group'),
        repeats=False,
        a_legacy=LegacyDefinition(name='x_fhi_aims_section_eigenvalues_group'))

    x_fhi_aims_energy_reference_fermi = Quantity(
        type=float, unit=ureg.eV, a_legacy=LegacyDefinition(name='x_fhi_aims_energy_reference_fermi'))


class section_atom_projected_dos(public.section_atom_projected_dos):

    m_def = Section(validate=False, extends_base_section=True, a_legacy=LegacyDefinition(name='section_atom_projected_dos'))

    x_fhi_aims_atom_projected_dos_file = Quantity(
        type=str,
        shape=[],
        description='''
        -
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_atom_projected_dos_file'))


class section_k_band(public.section_k_band):

    m_def = Section(validate=False, extends_base_section=True, a_legacy=LegacyDefinition(name='section_k_band'))

    x_fhi_aims_band_k1 = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        -
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_band_k1'))

    x_fhi_aims_band_k2 = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        -
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_band_k2'))

    x_fhi_aims_band_k3 = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        -
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_band_k3'))

    x_fhi_aims_band_occupations_eigenvalue_string = Quantity(
        type=str,
        shape=[],
        description='''
        -
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_band_occupations_eigenvalue_string'))

    x_fhi_aims_band_segment = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        -
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_band_segment'))


class section_method(public.section_method):

    m_def = Section(validate=False, extends_base_section=True, a_legacy=LegacyDefinition(name='section_method'))

    x_fhi_aims_controlIn_charge = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        -
        ''',
        categories=[x_fhi_aims_controlIn_method],
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlIn_charge'))

    x_fhi_aims_controlIn_hse_omega = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='1 / meter',
        description='''
        -
        ''',
        categories=[x_fhi_aims_controlIn_method, public.settings_potential_energy_surface, public.settings_XC, public.settings_XC_functional],
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlIn_hse_omega'))

    x_fhi_aims_controlIn_hse_unit = Quantity(
        type=str,
        shape=[],
        description='''
        -
        ''',
        categories=[x_fhi_aims_controlIn_method, public.settings_potential_energy_surface, public.settings_XC, public.settings_XC_functional],
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlIn_hse_unit'))

    x_fhi_aims_controlIn_hybrid_xc_coeff = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        -
        ''',
        categories=[x_fhi_aims_controlIn_method],
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlIn_hybrid_xc_coeff'))

    x_fhi_aims_controlIn_k1 = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        -
        ''',
        categories=[x_fhi_aims_controlIn_method],
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlIn_k1'))

    x_fhi_aims_controlIn_k2 = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        -
        ''',
        categories=[x_fhi_aims_controlIn_method],
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlIn_k2'))

    x_fhi_aims_controlIn_k3 = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        -
        ''',
        categories=[x_fhi_aims_controlIn_method],
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlIn_k3'))

    x_fhi_aims_controlIn_k_grid = Quantity(
        type=np.dtype(np.int32),
        shape=[3],
        description='''
        -
        ''',
        categories=[x_fhi_aims_controlIn_method],
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlIn_k_grid'))

    x_fhi_aims_controlIn_occupation_order = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        -
        ''',
        categories=[x_fhi_aims_controlIn_method],
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlIn_occupation_order'))

    x_fhi_aims_controlIn_occupation_type = Quantity(
        type=str,
        shape=[],
        description='''
        -
        ''',
        categories=[x_fhi_aims_controlIn_method],
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlIn_occupation_type'))

    x_fhi_aims_controlIn_occupation_width = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        -
        ''',
        categories=[x_fhi_aims_controlIn_method],
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlIn_occupation_width'))

    x_fhi_aims_controlIn_override_relativity = Quantity(
        type=str,
        shape=[],
        description='''
        -
        ''',
        categories=[x_fhi_aims_controlIn_method, public.settings_potential_energy_surface, public.settings_XC, public.settings_relativity],
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlIn_override_relativity'))

    x_fhi_aims_controlIn_relativistic_threshold = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        -
        ''',
        categories=[x_fhi_aims_controlIn_method, public.settings_potential_energy_surface, public.settings_XC, public.settings_relativity],
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlIn_relativistic_threshold'))

    x_fhi_aims_controlIn_relativistic = Quantity(
        type=str,
        shape=[],
        description='''
        -
        ''',
        categories=[x_fhi_aims_controlIn_method, public.settings_potential_energy_surface, public.settings_XC, public.settings_relativity],
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlIn_relativistic'))

    x_fhi_aims_controlIn_sc_accuracy_eev = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        -
        ''',
        categories=[x_fhi_aims_controlIn_method],
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlIn_sc_accuracy_eev'))

    x_fhi_aims_controlIn_sc_accuracy_etot = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        -
        ''',
        categories=[x_fhi_aims_controlIn_method],
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlIn_sc_accuracy_etot'))

    x_fhi_aims_controlIn_sc_accuracy_forces = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        -
        ''',
        categories=[x_fhi_aims_controlIn_method],
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlIn_sc_accuracy_forces'))

    x_fhi_aims_controlIn_sc_accuracy_rho = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        -
        ''',
        categories=[x_fhi_aims_controlIn_method],
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlIn_sc_accuracy_rho'))

    x_fhi_aims_controlIn_sc_accuracy_stress = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        -
        ''',
        categories=[x_fhi_aims_controlIn_method],
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlIn_sc_accuracy_stress'))

    x_fhi_aims_controlIn_sc_iter_limit = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        -
        ''',
        categories=[x_fhi_aims_controlIn_method],
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlIn_sc_iter_limit'))

    x_fhi_aims_controlIn_spin = Quantity(
        type=str,
        shape=[],
        description='''
        -
        ''',
        categories=[x_fhi_aims_controlIn_method],
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlIn_spin'))

    x_fhi_aims_controlIn_verbatim_writeout = Quantity(
        type=str,
        shape=[],
        description='''
        -
        ''',
        categories=[x_fhi_aims_controlIn_method],
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlIn_verbatim_writeout'))

    x_fhi_aims_controlIn_xc = Quantity(
        type=str,
        shape=[],
        description='''
        -
        ''',
        categories=[x_fhi_aims_controlIn_method, public.settings_potential_energy_surface, public.settings_XC, public.settings_XC_functional],
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlIn_xc'))

    x_fhi_aims_controlInOut_band_segment_end1 = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        -
        ''',
        categories=[x_fhi_aims_controlInOut_method],
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlInOut_band_segment_end1'))

    x_fhi_aims_controlInOut_band_segment_end2 = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        -
        ''',
        categories=[x_fhi_aims_controlInOut_method],
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlInOut_band_segment_end2'))

    x_fhi_aims_controlInOut_band_segment_end3 = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        -
        ''',
        categories=[x_fhi_aims_controlInOut_method],
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlInOut_band_segment_end3'))

    x_fhi_aims_controlInOut_band_segment_start1 = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        -
        ''',
        categories=[x_fhi_aims_controlInOut_method],
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlInOut_band_segment_start1'))

    x_fhi_aims_controlInOut_band_segment_start2 = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        -
        ''',
        categories=[x_fhi_aims_controlInOut_method],
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlInOut_band_segment_start2'))

    x_fhi_aims_controlInOut_band_segment_start3 = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        -
        ''',
        categories=[x_fhi_aims_controlInOut_method],
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlInOut_band_segment_start3'))

    x_fhi_aims_controlInOut_hse_omega = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='1 / meter',
        description='''
        -
        ''',
        categories=[x_fhi_aims_controlInOut_method, public.settings_XC, public.settings_potential_energy_surface, public.settings_XC_functional],
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlInOut_hse_omega'))

    x_fhi_aims_controlInOut_hse_unit = Quantity(
        type=str,
        shape=[],
        description='''
        -
        ''',
        categories=[x_fhi_aims_controlInOut_method, public.settings_XC, public.settings_potential_energy_surface, public.settings_XC_functional],
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlInOut_hse_unit'))

    x_fhi_aims_controlInOut_hybrid_xc_coeff = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        -
        ''',
        categories=[x_fhi_aims_controlInOut_method],
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlInOut_hybrid_xc_coeff'))

    x_fhi_aims_controlInOut_k1 = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        -
        ''',
        categories=[x_fhi_aims_controlInOut_method],
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlInOut_k1'))

    x_fhi_aims_controlInOut_k2 = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        -
        ''',
        categories=[x_fhi_aims_controlInOut_method],
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlInOut_k2'))

    x_fhi_aims_controlInOut_k3 = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        -
        ''',
        categories=[x_fhi_aims_controlInOut_method],
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlInOut_k3'))

    x_fhi_aims_controlInOut_k_grid = Quantity(
        type=np.dtype(np.int32),
        shape=[3],
        description='''
        -
        ''',
        categories=[x_fhi_aims_controlInOut_method],
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlInOut_k_grid'))

    x_fhi_aims_controlInOut_number_of_spin_channels = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        -
        ''',
        categories=[x_fhi_aims_controlInOut_method],
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlInOut_number_of_spin_channels'))

    x_fhi_aims_controlInOut_override_relativity = Quantity(
        type=str,
        shape=[],
        description='''
        -
        ''',
        categories=[x_fhi_aims_controlInOut_method, public.settings_XC, public.settings_potential_energy_surface, public.settings_relativity],
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlInOut_override_relativity'))

    x_fhi_aims_controlInOut_relativistic_threshold = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        -
        ''',
        categories=[x_fhi_aims_controlInOut_method, public.settings_potential_energy_surface, public.settings_XC, public.settings_relativity],
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlInOut_relativistic_threshold'))

    x_fhi_aims_controlInOut_relativistic = Quantity(
        type=str,
        shape=[],
        description='''
        -
        ''',
        categories=[x_fhi_aims_controlInOut_method, public.settings_potential_energy_surface, public.settings_XC, public.settings_relativity],
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlInOut_relativistic'))

    x_fhi_aims_controlInOut_xc = Quantity(
        type=str,
        shape=[],
        description='''
        -
        ''',
        categories=[x_fhi_aims_controlInOut_method, public.settings_XC, public.settings_potential_energy_surface, public.settings_XC_functional],
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlInOut_xc'))

    x_fhi_aims_section_controlIn_basis_set = SubSection(
        sub_section=SectionProxy('x_fhi_aims_section_controlIn_basis_set'),
        repeats=True,
        a_legacy=LegacyDefinition(name='x_fhi_aims_section_controlIn_basis_set'))

    x_fhi_aims_section_MD_detect = SubSection(
        sub_section=SectionProxy('x_fhi_aims_section_MD_detect'),
        repeats=True,
        a_legacy=LegacyDefinition(name='x_fhi_aims_section_MD_detect'))


class section_run(public.section_run):

    m_def = Section(validate=False, extends_base_section=True, a_legacy=LegacyDefinition(name='section_run'))

    x_fhi_aims_controlIn_MD_time_step = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='second',
        description='''
        -
        ''',
        categories=[public.settings_run, x_fhi_aims_controlIn_run],
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlIn_MD_time_step'))

    x_fhi_aims_controlInOut_MD_time_step = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='second',
        description='''
        -
        ''',
        categories=[public.settings_run, x_fhi_aims_controlInOut_run],
        a_legacy=LegacyDefinition(name='x_fhi_aims_controlInOut_MD_time_step'))

    x_fhi_aims_geometry_optimization_converged = Quantity(
        type=str,
        shape=[],
        description='''
        Determines whether a geoemtry optimization is converged.
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_geometry_optimization_converged'))

    x_fhi_aims_number_of_tasks = Quantity(
        type=str,
        shape=[],
        description='''
        -
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_number_of_tasks'))

    x_fhi_aims_program_compilation_date = Quantity(
        type=str,
        shape=[],
        description='''
        -
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_program_compilation_date'))

    x_fhi_aims_program_compilation_time = Quantity(
        type=str,
        shape=[],
        description='''
        -
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_program_compilation_time'))

    x_fhi_aims_program_execution_date = Quantity(
        type=str,
        shape=[],
        description='''
        -
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_program_execution_date'))

    x_fhi_aims_program_execution_time = Quantity(
        type=str,
        shape=[],
        description='''
        -
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_program_execution_time'))

    x_fhi_aims_section_parallel_tasks = SubSection(
        sub_section=SectionProxy('x_fhi_aims_section_parallel_tasks'),
        repeats=False,
        a_legacy=LegacyDefinition(name='x_fhi_aims_section_parallel_tasks'))


class section_dos(public.section_dos):

    m_def = Section(validate=False, extends_base_section=True, a_legacy=LegacyDefinition(name='section_dos'))

    x_fhi_aims_dos_energy = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='joule',
        description='''
        -
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_dos_energy'))

    x_fhi_aims_dos_value_string = Quantity(
        type=str,
        shape=[],
        description='''
        -
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_dos_value_string'))


class section_system(public.section_system):

    m_def = Section(validate=False, extends_base_section=True, a_legacy=LegacyDefinition(name='section_system'))

    x_fhi_aims_geometry_atom_labels = Quantity(
        type=str,
        shape=[],
        description='''
        labels of atom
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_geometry_atom_labels'))

    x_fhi_aims_geometry_atom_positions_x = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='meter',
        description='''
        x component of atomic position
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_geometry_atom_positions_x'))

    x_fhi_aims_geometry_atom_positions_y = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='meter',
        description='''
        y component of atomic position
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_geometry_atom_positions_y'))

    x_fhi_aims_geometry_atom_positions_z = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='meter',
        description='''
        z component of atomic position
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_geometry_atom_positions_z'))

    x_fhi_aims_geometry_atom_velocity_x = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='meter / second',
        description='''
        x component of atomic velocity
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_geometry_atom_velocity_x'))

    x_fhi_aims_geometry_atom_velocity_y = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='meter / second',
        description='''
        y component of atomic velocity
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_geometry_atom_velocity_y'))

    x_fhi_aims_geometry_atom_velocity_z = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='meter / second',
        description='''
        z component of atomic velocity
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_geometry_atom_velocity_z'))

    x_fhi_aims_geometry_lattice_vector_x = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='meter',
        description='''
        x component of lattice vector
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_geometry_lattice_vector_x'))

    x_fhi_aims_geometry_lattice_vector_y = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='meter',
        description='''
        y component of lattice vector
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_geometry_lattice_vector_y'))

    x_fhi_aims_geometry_lattice_vector_z = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='meter',
        description='''
        z component of lattice vector
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_geometry_lattice_vector_z'))


class section_atom_type(common.section_atom_type):

    m_def = Section(validate=False, extends_base_section=True, a_legacy=LegacyDefinition(name='section_atom_type'))

    x_fhi_aims_section_controlInOut_atom_species = SubSection(
        sub_section=SectionProxy('x_fhi_aims_section_controlInOut_atom_species'),
        repeats=True,
        a_legacy=LegacyDefinition(name='x_fhi_aims_section_controlInOut_atom_species'))


class section_species_projected_dos(public.section_species_projected_dos):

    m_def = Section(validate=False, extends_base_section=True, a_legacy=LegacyDefinition(name='section_species_projected_dos'))

    x_fhi_aims_species_projected_dos_file = Quantity(
        type=str,
        shape=[],
        description='''
        -
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_species_projected_dos_file'))

    x_fhi_aims_species_projected_dos_species_label = Quantity(
        type=str,
        shape=[],
        description='''
        -
        ''',
        a_legacy=LegacyDefinition(name='x_fhi_aims_species_projected_dos_species_label'))


m_package.__init_metainfo__()
