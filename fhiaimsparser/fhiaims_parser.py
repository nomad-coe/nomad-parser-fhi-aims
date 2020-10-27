import os
import numpy as np
import pint
import logging

from .metainfo import m_env
from nomad.parsing.parser import FairdiParser

from nomad.parsing.text_parser import UnstructuredTextFileParser, Quantity, ParsePattern,\
    DataTextFileParser

from nomad.datamodel.metainfo.public import section_run, section_method, section_system,\
    section_XC_functionals, section_scf_iteration, section_single_configuration_calculation,\
    section_sampling_method, section_frame_sequence, section_eigenvalues, section_dos,\
    section_atom_projected_dos, section_species_projected_dos, section_k_band,\
    section_k_band_segment, section_energy_van_der_Waals, section_calculation_to_calculation_refs,\
    section_method_to_method_refs
from nomad.datamodel.metainfo.common import section_topology, section_atom_type
from .metainfo.fhi_aims import section_run as xsection_run, section_method as xsection_method,\
    x_fhi_aims_section_parallel_task_assignement, x_fhi_aims_section_parallel_tasks,\
    x_fhi_aims_section_controlIn_basis_set, x_fhi_aims_section_controlIn_basis_func,\
    x_fhi_aims_section_controlInOut_atom_species, x_fhi_aims_section_controlInOut_basis_func,\
    x_fhi_aims_section_eigenvalues_group, x_fhi_aims_section_eigenvalues_spin,\
    x_fhi_aims_section_eigenvalues_list_perturbativeGW, x_fhi_aims_section_vdW_TS


class FHIAimsControlParser(UnstructuredTextFileParser):
    def __init__(self):
        super().__init__(None)

    @staticmethod
    def str_to_unit(val_in):
        val = val_in.strip().lower()
        unit = None
        if val.startswith('a'):
            unit = '1/angstrom'
        elif val.startswith('b'):
            unit = '1/bohr'
        return unit

    def init_quantities(self):
        self._quantities = []

        self._quantities.append(
            Quantity(
                xsection_method.x_fhi_aims_controlIn_charge,
                r'\n *charge\s*([\d\-\+Ee\.]+)', repeats=False)
        )

        self._quantities.append(
            Quantity(
                xsection_method.x_fhi_aims_controlIn_hse_unit,
                r'\n *hse_unit\s*([\w\-]+)', str_operation=self.str_to_unit, repeats=False)
        )

        self._quantities.append(
            Quantity(
                xsection_method.x_fhi_aims_controlIn_hybrid_xc_coeff,
                r'\n *hybrid_xc_coeff\s*([\d\-\+Ee\.]+)', repeats=False)
        )

        self._quantities.append(
            Quantity(
                xsection_run.x_fhi_aims_controlIn_MD_time_step,
                r'\n *MD_time_step\s*([\d\-\+Ee\.]+)', repeats=False)
        )

        self._quantities.append(
            Quantity(
                xsection_method.x_fhi_aims_controlIn_k_grid,
                r'\n *k\_grid\s*([\d ]+)', repeats=False)
        )

        self._quantities.append(
            Quantity(
                'occupation_type', r'\n *occupation_type\s*([\w\. \-\+]+)', repeats=False)
        )

        self._quantities.append(
            Quantity(
                xsection_method.x_fhi_aims_controlIn_override_relativity,
                r'\n *override_relativity\s*([\.\w]+)', repeats=False)
        )

        self._quantities.append(
            Quantity(
                'relativistic', r'\n *relativistic\s*([\w\. \-\+]+)', repeats=False)
        )

        self._quantities.append(
            Quantity(
                xsection_method.x_fhi_aims_controlIn_sc_accuracy_rho,
                r'\n *sc_accuracy_rho\s*([\d\-\+Ee\.]+)', repeats=False)
        )

        self._quantities.append(
            Quantity(
                xsection_method.x_fhi_aims_controlIn_sc_accuracy_eev,
                r'\n *sc_accuracy_eev\s*([\d\-\+Ee\.]+)', repeats=False)
        )

        self._quantities.append(
            Quantity(
                xsection_method.x_fhi_aims_controlIn_sc_accuracy_etot,
                r'\n *sc_accuracy_etot\s*([\d\-\+Ee\.]+)', repeats=False)
        )

        self._quantities.append(
            Quantity(
                xsection_method.x_fhi_aims_controlIn_sc_accuracy_forces,
                r'\n *sc_accuracy_forces\s*([\d\-\+Ee\.]+)', repeats=False)
        )

        self._quantities.append(
            Quantity(
                xsection_method.x_fhi_aims_controlIn_sc_accuracy_stress,
                r'\n *sc_accuracy_stress\s*([\d\-\+Ee\.]+)', repeats=False)
        )

        self._quantities.append(
            Quantity(
                xsection_method.x_fhi_aims_controlIn_sc_iter_limit,
                r'\n *sc_iter_limit\s*([\d]+)', repeats=False)
        )

        self._quantities.append(
            Quantity(
                xsection_method.x_fhi_aims_controlIn_spin,
                r'\n *spin\s*([\w]+)', repeats=False)
        )

        self._quantities.append(
            Quantity(
                xsection_method.x_fhi_aims_controlIn_verbatim_writeout,
                r'\n *verbatim_writeout\s*([\w]+)', repeats=False)
        )

        self._quantities.append(
            Quantity(
                'xc', r'\n *xc\s*([\w\. \-\+]+)', repeats=False)
        )

        def str_to_species(val_in):
            val = val_in.strip().split('\n')
            data = []
            species = dict()
            for v in val:
                v = v.strip()
                if v.startswith('#') or not v or not v[0].isalpha():
                    continue
                if v.startswith('species'):
                    if species:
                        data.append(species)
                    species = dict(species=v.split()[1:])
                else:
                    v = v.replace('.d', '.e').split()
                    vi = v[1] if len(v[1:]) == 1 else v[1:]
                    if v[0] in species:
                        species[v[0]].extend([vi])
                    else:
                        species[v[0]] = [vi]
            data.append(species)
            return data

        self._quantities.append(
            Quantity(
                'species', r'\n *(species\s*[A-Z][a-z]?[\s\S]+)'
                r'(gaussian|hydro|valence|ion_occ|ionic|confined)([\w ]*)\n',
                str_operation=str_to_species, repeats=False)
        )


class FHIAimsOutParser(UnstructuredTextFileParser):
    def __init__(self):
        super().__init__(None)

    def init_quantities(self):
        self._quantities = []
        self._quantities.append(
            Quantity(
                section_run.program_version, ParsePattern(
                    head=r'Invoking FHI\-aims', key=r'(?:Version|FHI-aims version)',
                    value=r'[\d\.]+', tail='\n'),
                repeats=False))

        self._quantities.append(
            Quantity(
                xsection_run.x_fhi_aims_program_compilation_date, ParsePattern(
                    head='Compiled', key='on', value=r'[\d\/ ]+', tail=' '),
                repeats=False)
        )

        self._quantities.append(
            Quantity(
                xsection_run.x_fhi_aims_program_compilation_time, ParsePattern(
                    head='Compiled', key='at', value=r'[\d\: ]+', tail=' '),
                repeats=False)
        )

        self._quantities.append(
            Quantity(
                section_run.program_compilation_host, ParsePattern(
                    head='Compiled', key='on host', value=r'[\w \.]+', tail=r'reporting\.\n'),
                repeats=False)
        )

        self._quantities.append(
            Quantity(
                xsection_run.x_fhi_aims_program_execution_date, r'Date\s*:\s*([0-9]+)',
                repeats=False)
        )

        self._quantities.append(
            Quantity(
                xsection_run.x_fhi_aims_program_execution_time, r'Time\s*:\s*([0-9\.]+)\n',
                repeats=False)
        )

        self.quantities.append(
            Quantity(
                section_run.time_run_cpu1_start,
                r'Time zero on CPU 1\s*:\s*([0-9\-E\.]+)\s*(?P<__unit>\w+)\.',
                repeats=False)
        )

        self.quantities.append(
            Quantity(
                section_run.time_run_wall_start,
                r'Internal wall clock time zero\s*:\s*([0-9\-E\.]+)\s*(?P<__unit>\w+)\.',
                repeats=False)
        )

        self._quantities.append(
            Quantity(section_run.raw_id, r'aims_uuid\s*:\s*([\w\-]+)', repeats=False)
        )

        self._quantities.append(
            Quantity(
                xsection_run.x_fhi_aims_number_of_tasks, r'Using\s*(\d+)\s*parallel tasks',
                repeats=False)
        )

        self._quantities.append(
            Quantity(
                x_fhi_aims_section_parallel_task_assignement.x_fhi_aims_parallel_task_nr,
                r'Task\s*(\d+)\s*on host', repeats=True)
        )

        self._quantities.append(
            Quantity(
                x_fhi_aims_section_parallel_task_assignement.x_fhi_aims_parallel_task_host,
                r'Task\s*\d+\s*on host\s*([\w\. ]+)reporting', repeats=True)
        )

        self._quantities.append(
            Quantity('fhi_aims_files', r'(?:FHI\-aims file:|Parsing)\s*([\w\/\.]+)', repeats=True)
        )

        def str_to_array_size_parameters(val_in):
            val = [v.lstrip(' |').split(':') for v in val_in.strip().split('\n')]
            return {v[0].strip(): int(v[1]) for v in val if len(v) == 2}

        self._quantities.append(
            Quantity(
                'array_size_parameters',
                r'Basic array size parameters:\s*([\|:\s\w\.\/]+:\s*\d+)', repeats=False,
                str_operation=str_to_array_size_parameters)
        )

        self._quantities.append(
            Quantity(
                xsection_method.x_fhi_aims_controlInOut_hse_unit,
                r'\n *hse_unit: Unit for the HSE06 hybrid functional screening parameter set to\s*(\w)',
                str_operation=FHIAimsControlParser.str_to_unit, repeats=False)
        )

        self._quantities.append(
            Quantity(
                xsection_method.x_fhi_aims_controlInOut_hybrid_xc_coeff,
                r'\n *hybrid_xc_coeff: Mixing coefficient for hybrid-functional exact exchange modified to\s*([\d\.]+)',
                repeats=False)
        )

        self._quantities.append(
            Quantity(
                xsection_method.x_fhi_aims_controlInOut_k_grid,
                r'\n *Found k-point grid:\s*([\d ]+)', repeats=False)
        )

        self._quantities.append(
            Quantity(
                xsection_run.x_fhi_aims_controlInOut_MD_time_step,
                r'\n *Molecular dynamics time step\s*=\s*([\d\.]+)\s*(?P<__unit>[\w]+)',
                repeats=False)
        )

        self._quantities.append(
            Quantity(
                xsection_method.x_fhi_aims_controlInOut_relativistic,
                r'\n *Scalar relativistic treatment of kinetic energy:\s*([\w]+)',
                repeats=False)
        )

        self._quantities.append(
            Quantity(
                xsection_method.x_fhi_aims_controlInOut_relativistic,
                r'\n *(Non-relativistic) treatment of kinetic energy',
                repeats=False)
        )

        self._quantities.append(
            Quantity(
                xsection_method.x_fhi_aims_controlInOut_relativistic_threshold,
                r'\n *Threshold value for ZORA:\s*([\d\.Ee\-\+])', repeats=False)
        )

        self._quantities.append(
            Quantity(
                xsection_method.x_fhi_aims_controlInOut_xc,
                r'\n *XC:\s*(?:Using)*\s*([\w\- ]+) with OMEGA =\s*([\d\.Ee\-\+]+)',
                repeats=False)
        )

        self._quantities.append(
            Quantity(
                xsection_method.x_fhi_aims_controlInOut_xc,
                r'XC: (?:Running|Using) ([\-\w \(\) ]+)', repeats=False)
        )

        self._quantities.append(
            Quantity(
                xsection_method.x_fhi_aims_controlInOut_xc,
                r'\n *(Hartree-Fock) calculation starts \.\.\.', repeats=False)
        )

        self._quantities.append(
            Quantity(
                'band_segment_points',
                r'Plot band\s*[\d]+\s*\|\s*begin([ \d\.]+)\s*\|\s*end([ \d\.]+)\s*\|\s*number of points:([ \d]+)',
                repeats=True)
        )

        def _str_to_species(val_in):
            val = [v.strip() for v in val_in.split('\n')]
            data = []
            species = dict()
            for i in range(len(val)):
                if val[i].startswith('Reading configuration options for species'):
                    if species:
                        data.append(species)
                    species = dict(species=val[i].split('species')[1].split()[0])
                elif not val[i].startswith('| Found'):
                    continue
                val[i] = val[i].split(':')
                if len(val[i]) < 2:
                    continue
                k = val[i][0].split('Found')[1].strip()
                v = val[i][1].replace(',', '').split()
                if 'Gaussian basis function' in k and 'elementary' in v:
                    n_gaussians = int(v[v.index('elementary') - 1])
                    for j in range(n_gaussians):
                        v.extend(val[i + j + 1].lstrip('|').split())
                v = v[0] if len(v) == 1 else v
                if val[i][0] in species:
                    species[k].extend([v])
                else:
                    species[k] = [v]
            data.append(species)
            return data

        self._quantities.append(
            Quantity(
                'species',
                r'(Reading configuration options for species [\s\S]+?)\n *Finished',
                str_operation=_str_to_species, repeats=False)
        )

        def str_to_species(val_in):
            data = dict()
            val = [v.strip() for v in val_in.split('\n')]
            for i in range(len(val)):
                if val[i].startswith('species'):
                    data['species'] = val[i].split()[1]
                elif not val[i].startswith('| Found'):
                    continue
                val[i] = val[i].split(':')
                if len(val[i]) < 2:
                    continue
                k = val[i][0].split('Found')[1].strip()
                v = val[i][1].replace(',', '').split()
                if 'Gaussian basis function' in k and 'elementary' in v:
                    n_gaussians = int(v[v.index('elementary') - 1])
                    for j in range(n_gaussians):
                        v.extend(val[i + j + 1].lstrip('|').split())
                v = v[0] if len(v) == 1 else v
                if val[i][0] in data:
                    data[k].extend([v])
                else:
                    data[k] = [v]
            return data

        self._quantities.append(
            Quantity(
                'control_inout',
                r'\n *Reading file control\.in\.\s*\-*\s*([\s\S]+?)'
                r'Finished reading input file \'control\.in\'', repeats=False,
                sub_parser=UnstructuredTextFileParser(quantities=[
                    Quantity(
                        'species', r'Reading configuration options for (species[\s\S]+?)grid points\.',
                        repeats=True, str_operation=str_to_species)]))
        )

        # The assign the initial geometry to full scf as no change in structure is done
        # during the initial scf step
        self._quantities.append(
            Quantity(
                'lattice_vectors',
                r'Input geometry:\s*\|\s*Unit cell:\s*'
                r'\s*\|\s*([\d\.\-\+eE\s]+)\s*\|\s*([\d\.\-\+eE\s]+)\s*\|\s*([\d\.\-\+eE\s]+)',
                repeats=False, unit='angstrom', shape=(3, 3), dtype=float)
        )

        def str_to_labels_positions(val_in):
            val = [v.split() for v in val_in.strip().split('\n')]
            val = [v for v in val if len(v) >= 4]
            velocities = [
                val.pop(i)[1:] for i in range(
                    len(val) - 1, -1, -1) if val[i][0].startswith('velocity')]

            # address different formatting by getting index of symbol and atom positions
            label_index = None
            position_index = None
            for i in range(len(val[0])):
                if val[0][i].isalpha() and 1 <= len(val[0][i]) <= 2 and val[0][i].istitle():
                    label_index = i
                try:
                    if position_index is None:
                        np.array(val[0][i:i + 3], dtype=float)
                        position_index = i
                except Exception:
                    continue

            if label_index is None or position_index is None:
                return

            labels = [v[label_index] for v in val if 'atom' in v or 'Species' in v]
            positions = np.array(
                [v[position_index:position_index + 3] for v in val if 'atom' in v or 'Species' in v],
                dtype=float)

            if velocities:
                return [labels, positions, velocities]

            return [labels, positions]

        self._quantities.append(
            Quantity(
                'labels_positions',
                r'Input geometry:\s*\|\s*(?:Unit cell:[\d\.\sEe\-\+\|]+|No unit cell requested\.\n)'
                r'\s*\|\s*Atomic structure:\s*\|\s*Atom\s*x \[A\]\s*y \[A\]\s*z \[A\]\n'
                r'([\s\S]*\s*\|\s*\d+:\s*Species\s*[A-Z][a-z]?\s*[\d\. \-]+\n)+',
                repeats=False, str_operation=str_to_labels_positions)
        )

        def str_to_eigenvalues(val_in):
            val = [v.strip() for v in val_in.strip().split('\n') if v]
            if val[0].startswith('K-point'):
                kpt = np.array(val[0].split()[3:6], dtype=float)
            else:
                kpt = None

            eigs = np.array([v.split() for v in val if v[0].isdecimal()], dtype=float)

            eigs = np.transpose(eigs)
            eig, occ = eigs[1:3]
            return kpt, eig, occ

        def str_to_energy_components(val_in):
            val = [v.strip() for v in val_in.strip().split('\n')]
            res = dict()
            for v in val:
                v = v.lstrip(' |').strip().split(':')
                if len(v) < 2 or not v[1]:
                    continue
                vi = v[1].split()
                if not vi[0][-1].isdecimal():
                    continue
                unit = None
                if len(vi) > 2:
                    unit = 'hartree' if vi[1] == 'Ha' else 'eV'
                res[v[0].strip()] = pint.Quantity(
                    float(vi[0]), unit) if unit is not None else float(vi[0])
            return res

        def str_to_scf_convergence(val_in):
            res = dict()
            for v in val_in.strip().split('\n'):
                v = v.lstrip(' | Change of').split(':')
                if len(v) != 2:
                    break
                res[v[0].strip()] = float(v[1].split()[0])
                return res

        def str_to_atomic_forces(val_in):
            val = [v.lstrip(' |').split() for v in val_in.strip().split('\n')]
            forces = np.array([v[1:4] for v in val if len(v) == 4], dtype=float)
            return pint.Quantity(forces, 'eV/angstrom')

        def str_to_dos_files(val_in):
            val = [v.strip() for v in val_in.strip().split('\n')]
            files = []
            species = []
            for v in val[1:]:
                if v.startswith('| writing') and 'raw data' in v:
                    files.append(v.split('to file')[1].strip(' .'))
                    if 'for species' in v:
                        species.append(v.split('for species')[1].split()[0])
                elif not v.startswith('|'):
                    break
            return files, list(set(species))

        def str_to_gw_eigs(val_in):
            val = [v.split() for v in val_in.split('\n')]
            keys = val[0]
            data = []
            for v in val[1:]:
                if len(keys) == len(v) and v[0].isdecimal():
                    data.append(v)
            data = np.transpose(data)
            res = {keys[i]: data[i] for i in range(len(data))}
            return res

        def str_to_gw_scf(val_in):
            val = [v.split(':') for v in val_in.split('\n')]
            data = {}
            for v in val:
                if len(v) == 2:
                    data[v[0].strip(' |')] = pint.Quantity(float(v[1].split()[0]), 'eV')
                if 'Fit accuracy for G' in v[0]:
                    data['Fit accuracy for G(w)'] = v[0].split()[-1]
            return data

        scf_quantities = Quantity(
            'self_consistency',
            r'Begin self\-consistency iteration #\s*\d+([\s\S]+?Total energy evaluation[s:\d\. ]+)',
            repeats=True, sub_parser=UnstructuredTextFileParser(quantities=[
                Quantity(
                    'date_time', r'\n *Date\s*:(\s*\d+), Time\s*:(\s*[\d\.]+)\s*', repeats=False),
                # section_eigenvalues is written on scc but for exciting it belongs to scf
                Quantity(
                    'eigenvalues',
                    r'(?:Spin-(?:up|down) eigenvalues:)*'
                    r'(K-point:\s*\d+ at\s* [\d\.\-\+ ]+\(in units of recip\. lattice\))*'
                    r'(\s*State\s*Occupation\s*Eigenvalue \[Ha\]\s*Eigenvalue \[eV\]\s*[\d\.\s\-Ee]+)',
                    repeats=True, str_operation=str_to_eigenvalues),
                Quantity(
                    'energy_components',
                    r'\n *Total energy components:([\s\S]+?)(\| Electronic free energy per atom[eV:\d\. ]+)',
                    repeats=False, str_operation=str_to_energy_components),
                # TODO include scf forces, stress, pressure in metainfo
                # Quantity(
                #         'atomic_forces', r'\n *Total forces\([\s\d]+\)\s*:([\s\d\.\-\+Ee]+)\n', repeats=True),
                # Quantity(
                #     'stress_tensor', r'\n *Sum of all contributions\s*:\s*([\d\.\-\+Ee ]+\n)', repeats=False),
                # Quantity(
                #     'pressure', r' *\|\s*Pressure:\s*([\d\.\-\+Ee ]+)', repeats=False),
                Quantity(
                    'scf_convergence',
                    r'\n *Self-consistency convergence accuracy:([\s\S]+?)(\| Change of total energy\s*:\s*[\d\.\-\+Ee]+)',
                    repeats=False, str_operation=str_to_scf_convergence),
                Quantity(
                    'humo', r'Highest occupied state \(VBM\) at\s*([\d\.\-\+Ee ]+) (?P<__unit>\w+)',
                    repeats=False),
                Quantity(
                    'lumo', r'Lowest unoccupied state \(CBM\) at\s*([\d\.\-\+Ee ]+) (?P<__unit>\w+)',
                    repeats=False),
                Quantity(
                    'fermi_level', r'\n *\| Chemical potential \(Fermi level\) in (?P<__unit>\w+)\s*:([\d\.\-\+Ee ]+)',
                    repeats=False)]))

        scf_gw_quantities = Quantity(
            'gw_self_consistency', r'GW Total Energy Calculation([\s\S]+?)\-{5}',
            repeats=True, str_operation=str_to_gw_scf)

        def str_to_hirshfeld(val_in):
            val = [v.strip() for v in val_in.strip().split('\n')]
            data = dict(atom=val[0])
            for v in val[1:]:
                if v.startswith('|'):
                    v = v.strip(' |').split(':')
                    if v[0][0].isalpha():
                        key = v[0]
                        data[key] = []
                    data[key].extend(v[-1].split())
            return data

        calculation_quantities = [
            Quantity(
                'date_time', r'\n *Date\s*:(\s*\d+), Time\s*:(\s*[\d\.]+)\s*', repeats=False),
            Quantity(
                'labels_positions',
                r'\n *(atom\s*[\s\S]+?)\n\-{5}', str_operation=str_to_labels_positions, repeats=False),
            Quantity(
                'lattice_vectors',
                r'\n *lattice_vector([\d\.\- ]+)\n *lattice_vector([\d\.\- ]+)\n *lattice_vector([\d\.\- ]+)',
                unit='angstrom', repeats=False, shape=(3, 3), dtype=float),
            Quantity(
                'energy',
                r'\n *Energy and forces in a compact form:([\s\S]+?Electronic free energy\s*:\s*[\d\.\-\+Ee] eV)',
                str_operation=str_to_energy_components, repeats=False),
            # in some cases, the energy components are also printed for after a calculation
            # same format as in scf iteration, they are printed also in initialization
            # so we should get last occurence
            Quantity(
                'energy_components',
                r'\n *Total energy components:([\s\S]*?)(\| Electronic free energy per atom[eV:\d\. ]+)',
                repeats=True, str_operation=str_to_energy_components),
            Quantity(
                'energy_xc',
                r'\n *Start decomposition of the XC Energy([\s\S]+?)End decomposition of the XC Energy',
                str_operation=str_to_energy_components, repeats=False),
            Quantity(
                'forces', r'\n *Total atomic forces.*?\[eV/Ang\]:\s*([\d\.Ee\-\+\s\|]+)',
                str_operation=str_to_atomic_forces, repeats=False),
            Quantity(
                'time_force_evaluation',
                r'\n *\| Time for this force evaluation\s*:\s*([\d\.]+) s\s*([\d\.]+) s', repeats=False),
            Quantity(
                'total_dos_files', r'Calculating total density of states([\s\S]+?)\-{5}',
                str_operation=str_to_dos_files, repeats=False),
            Quantity(
                'atom_projected_dos_files', r'Calculating atom\-projected density of states([\s\S]+?)\-{5}',
                str_operation=str_to_dos_files, repeats=False),
            Quantity(
                'species_projected_dos_files', r'Calculating angular momentum projected density of states([\s\S]+?)\-{5}',
                str_operation=str_to_dos_files, repeats=False),
            Quantity(
                'gw_eigenvalues', r'(state\s*occ_num\s*e_gs[\s\S]+?)\s*\| Total time',
                str_operation=str_to_gw_eigs, repeats=False),
            Quantity(
                'vdW_TS',
                r'Evaluating non-empirical van der Waals correction[\s\S]+?\|\s*Converged\.',
                repeats=False, sub_parser=UnstructuredTextFileParser(quantities=[
                    Quantity(
                        'kind', r'Evaluating non-empirical van der Waals correction \(([\w/]+)\)',
                        repeats=False),
                    Quantity(
                        'atom_hirshfeld', r'\| Atom\s*\d+:[\s\S]+?\-{5}',
                        str_operation=str_to_hirshfeld, repeats=True)])),
            Quantity('converged', r'Self\-consistency cycle (converged)\.', repeats=False),
            scf_quantities,
            scf_gw_quantities
        ]

        self._quantities.append(
            Quantity(
                'full_scf',
                r'Begin self-consistency loop: Initialization'
                r'([\s\S]+?(?:Time for this force evaluation\s*:\s*[s \d\.]+|Final output of selected total energy values))',
                repeats=True, sub_parser=UnstructuredTextFileParser(quantities=calculation_quantities))
        )

        self._quantities.append(
            Quantity(
                'geometry_optimization',
                r'\n *Geometry optimization: Attempting to predict improved coordinates\.'
                r'([\s\S]+?(?:Time for this force evaluation\s*:\s*[s \d\.]+|Final output of selected total energy values))',
                repeats=True, sub_parser=UnstructuredTextFileParser(quantities=calculation_quantities))
        )

        self._quantities.append(
            Quantity(
                'molecular_dynamics',
                r'\n *Molecular dynamics: Attempting to update all nuclear coordinates\.'
                r'([\s\S]+?(?:Time for this force evaluation\s*:\s*[s \d\.]+|Final output of selected total energy values))',
                repeats=True, sub_parser=UnstructuredTextFileParser(quantities=calculation_quantities))
        )

        # TODO add SOC perturbed eigs

    def get_number_of_spin_channels(self):
        array_size = self.get('array_size_parameters')
        return array_size.get('Number of spin channels', 1)

    def get_calculation_type(self):
        calculation_type = 'geometry_optimization'
        if self.get('molecular_dynamics', None) is not None:
            calculation_type = 'molecular_dynamics'
        # TODO there should be a single point calculation type
        return calculation_type


class FHIAimsParser(FairdiParser):
    def __init__(self):
        super().__init__(
            name='parsers/fhi-aims', code_name='FHI-aims',
            code_homepage='https://aimsclub.fhi-berlin.mpg.de/',
            mainfile_contents_re=(
                r'^(.*\n)*'
                r'?\s*Invoking FHI-aims \.\.\.'))
        self._metainfo_env = m_env

        self.out_parser = FHIAimsOutParser()
        self.control_parser = FHIAimsControlParser()
        self.dos_parser = DataTextFileParser()
        self.bandstructure_parser = DataTextFileParser()

        self._xc_map = {
            'Perdew-Wang parametrisation of Ceperley-Alder LDA': [
                {'name': 'LDA_C_PW'}, {'name': 'LDA_X'}],
            'Perdew-Zunger parametrisation of Ceperley-Alder LDA': [
                {'name': 'LDA_C_PZ'}, {'name': 'LDA_X'}],
            'VWN-LDA parametrisation of VWN5 form': [
                {'name': 'LDA_C_VWN'}, {'name': 'LDA_X'}],
            'VWN-LDA parametrisation of VWN-RPA form': [
                {'name': 'LDA_C_VWN_RPA'}, {'name': 'LDA_X'}],
            'AM05 gradient-corrected functionals': [
                {'name': 'GGA_C_AM05'}, {'name': 'GGA_X_AM05'}],
            'BLYP functional': [{'name': 'GGA_C_LYP'}, {'name': 'GGA_X_B88'}],
            'PBE gradient-corrected functionals': [
                {'name': 'GGA_C_PBE'}, {'name': 'GGA_X_PBE'}],
            'PBEint gradient-corrected functional': [
                {'name': 'GGA_C_PBEINT'}, {'name': 'GGA_X_PBEINT'}],
            'PBEsol gradient-corrected functionals': [
                {'name': 'GGA_C_PBE_SOL'}, {'name': 'GGA_X_PBE_SOL'}],
            'RPBE gradient-corrected functionals': [
                {'name': 'GGA_C_PBE'}, {'name': 'GGA_X_RPBE'}],
            'revPBE gradient-corrected functionals': [
                {'name': 'GGA_C_PBE'}, {'name': 'GGA_X_PBE_R'}],
            'PW91 gradient-corrected functionals': [
                {'name': 'GGA_C_PW91'}, {'name': 'GGA_X_PW91'}],
            'M06-L gradient-corrected functionals': [
                {'name': 'MGGA_C_M06_L'}, {'name': 'MGGA_X_M06_L'}],
            'M11-L gradient-corrected functionals': [
                {'name': 'MGGA_C_M11_L'}, {'name': 'MGGA_X_M11_L'}],
            'TPSS gradient-corrected functionals': [
                {'name': 'MGGA_C_TPSS'}, {'name': 'MGGA_X_TPSS'}],
            'TPSSloc gradient-corrected functionals': [
                {'name': 'MGGA_C_TPSSLOC'}, {'name': 'MGGA_X_TPSS'}],
            'hybrid B3LYP functional': [
                {'name': 'HYB_GGA_XC_B3LYP5'}],
            'Hartree-Fock': [{'name': 'HF_X'}],
            'HSE': [{'name': 'HYB_GGA_XC_HSE03'}],
            'HSE-functional': [{'name': 'HYB_GGA_XC_HSE06'}],
            'hybrid-PBE0 functionals': [
                {'name': 'GGA_C_PBE'}, {
                    'name': 'GGA_X_PBE', 'weight': lambda x: 1.0 - x},
                {'name': 'HF_X', 'weight': lambda x: x}],
            'hybrid-PBEsol0 functionals': [
                {'name': 'GGA_C_PBE_SOL'}, {
                    'name': 'GGA_X_PBE_SOL', 'weight': lambda x: 1.0 - x},
                {'name': 'HF_X', 'weight': lambda x: x}],
            'Hybrid M06 gradient-corrected functionals': [{'name': 'HYB_MGGA_XC_M06'}],
            'Hybrid M06-2X gradient-corrected functionals': [
                {'name': 'HYB_MGGA_XC_M06_2X'}],
            'Hybrid M06-HF gradient-corrected functionals': [
                {'name': 'HYB_MGGA_XC_M06_HF'}],
            'Hybrid M08-HX gradient-corrected functionals': [
                {'name': 'HYB_MGGA_XC_M08_HX'}],
            'Hybrid M08-SO gradient-corrected functionals': [
                {'name': 'HYB_MGGA_XC_M08_SO'}],
            'Hybrid M11 gradient-corrected functionals': [{'name': 'HYB_MGGA_XC_M11'}]}

        # TODO update metainfo to reflect all energy corrections
        # why section_vdW_TS under x_fhi_aims_section_controlInOut_atom_species?
        self._energy_map = {
            'Total energy uncorrected': 'energy_total',
            'Total energy corrected': 'energy_total_T0',
            'Electronic free energy': 'energy_free',
            'X Energy': 'energy_X',
            'C Energy GGA': 'energy_C',
            'Total XC Energy': 'energy_XC_functional',
            'X Energy LDA': 'x_fhi_aims_energy_X_LDA',
            'C Energy LDA': 'x_fhi_aims_energy_C_LDA',
            'Sum of eigenvalues': 'energy_sum_eigenvalues',
            'XC energy correction': 'energy_XC',
            'XC potential correction': 'energy_XC_potential',
            'Free-atom electrostatic energy': 'x_fhi_aims_energy_electrostatic_free_atom',
            'Hartree energy correction': 'energy_correction_hartree',
            'vdW energy correction': 'energy_van_der_Waals_error_scf_iteration',
            'Entropy correction': 'energy_correction_entropy',
            'Total energy': 'energy_total',
            'Total energy, T -> 0': 'energy_total_T0',
            'Kinetic energy': 'electronic_kinetic_energy',
            'Electrostatic energy': 'energy_electrostatic',
            'error in Hartree potential': 'energy_hartree_error',
            'Sum of eigenvalues per atom': 'energy_sum_eigenvalues_per_atom',
            'Total energy (T->0) per atom': 'energy_total_T0_per_atom',
            'Electronic free energy per atom': 'energy_free_per_atom',
            'Hartree-Fock part': 'energy_hartree_fock_X_scaled',
            # GW
            'Galitskii-Migdal Total Energy': 'x_fhi_aims_scgw_galitskii_migdal_total_energy',
            'GW Kinetic Energy': 'x_fhi_aims_scgw_kinetic_energy',
            'Hartree energy from GW density': 'x_fhi_aims_scgw_hartree_energy_sum_eigenvalues_scf_iteration',
            'GW correlation Energy': 'x_fhi_aims_energy_scgw_correlation_energy',
            'RPA correlation Energy': 'x_fhi_aims_scgw_rpa_correlation_energy',
            'Sigle Particle Energy': 'x_fhi_aims_single_particle_energy',
            'Fit accuracy for G(w)': 'x_fhi_aims_poles_fit_accuracy'
        }

    def get_fhiaims_file(self, default):
        base, ext = default.split('.')
        base = base.lower()
        files = os.listdir(self.maindir)
        files = [os.path.basename(f) for f in files]
        files = [os.path.join(
            self.maindir, f) for f in files if base.lower() in f.lower() and f.endswith(ext)]
        files.sort()
        return files

    def parse_configurations(self):
        sec_run = self.archive.section_run[-1]

        def parse_bandstructure():
            # band structure, unlike dos is not a property of a section_scc but of the
            # the whole run. dos output file is contained in a section
            sec_scc = sec_run.section_single_configuration_calculation[-1]
            bandstructure_files = self.get_fhiaims_file('band.out')
            if not bandstructure_files:
                return
            sec_k_band = sec_scc.m_create(section_k_band)
            sec_k_band.band_structure_kind = 'electronic'
            for band_file in bandstructure_files:
                self.bandstructure_parser.mainfile = band_file
                if self.bandstructure_parser.data is None:
                    continue
                data = np.transpose(self.bandstructure_parser.data)
                k_points = np.transpose(data[1:4])
                band_occupations = np.transpose(data[4::2])
                band_energies = np.transpose(data[5::2])
                sec_k_band_segment = sec_k_band.m_create(section_k_band_segment)
                sec_k_band_segment.band_k_points = k_points
                # TODO verify spin formatting
                sec_k_band_segment.band_occupations = [band_occupations]
                sec_k_band_segment.band_energies = pint.Quantity([band_energies], 'eV')

        def read_dos(dos_file):
            dos_file = self.get_fhiaims_file(dos_file)
            if not dos_file:
                return
            self.dos_parser.mainfile = dos_file[0]
            if self.dos_parser.data is None:
                return
            return np.transpose(self.dos_parser.data)

        def read_projected_dos(dos_files):
            dos = []
            dos_dn = []
            n_atoms = 0
            l_max = []
            for dos_file in dos_files:
                data = read_dos(dos_file)
                if data is None:
                    continue
                # we first read the spin_dn just to make sure the spin dn component exists
                if 'spin_up' in dos_file:
                    data_dn = read_dos(dos_file.replace('spin_up', 'spin_dn'))
                    if data_dn is None:
                        continue
                    # the maximum l is arbitrary for different species, so we cant stack
                    dos_dn.append(data_dn[1:])
                l_max.append(len(data[1:]))
                # column 0 is energy column 1 is total
                dos.append(data[1:])
                n_atoms += 1

            if not dos:
                return None, None
            energies = pint.Quantity(data[0], 'eV')
            n_l = min(l_max)
            n_spin = 2 if dos_dn else 1
            dos = [d[:n_l] for d in dos]
            if dos_dn:
                dos_dn = [d[:n_l] for d in dos_dn]
                dos = np.hstack((dos, dos_dn))
            dos = np.reshape(dos, (n_l, n_spin, n_atoms, len(energies)))
            dos = pint.Quantity(dos, '1/eV')
            return energies, dos

        def parse_dos(section):
            sec_scc = sec_run.section_single_configuration_calculation[-1]
            sec_dos = None
            energies = None
            lattice_vectors = section.get(
                'lattice_vectors', self.out_parser.get('lattice_vectors'))
            if lattice_vectors is None:
                lattice_vectors = pint.Quantity(np.eye(3))
            volume = np.abs(np.linalg.det(lattice_vectors.magnitude))
            n_spin = self.out_parser.get_number_of_spin_channels()
            # parse total first, we expect only one file
            total_dos_files, _ = section.get('total_dos_files', [['KS_DOS_total_raw.dat'], []])
            for dos_file in total_dos_files:
                data = read_dos(dos_file)
                if data is None:
                    continue
                sec_dos = sec_scc.m_create(section_dos)
                sec_dos.dos_kind = 'electronic'
                energies = pint.Quantity(data[0], 'eV')
                sec_dos.number_of_dos_values = len(energies)
                sec_dos.dos_energies = energies
                # dos unit is 1/(eV-cell volume)
                dos = pint.Quantity(data[1: n_spin + 1] * volume, 'angstrom**3/eV')
                sec_dos.dos_values = dos.to('m**3/J').magnitude

            # parse projected
            # projected does for different spins on separate files
            # we only include spin_up, add spin_dn in loop
            for projection_type in ['atom', 'species']:
                proj_dos_files, species = section.get(
                    '%s_projected_dos_files' % projection_type, [[], []])
                proj_dos_files = [f for f in proj_dos_files if 'spin_dn' not in f]
                if proj_dos_files:
                    energies, dos = read_projected_dos(proj_dos_files)
                    if dos is None:
                        continue
                    if projection_type == 'atom':
                        sec_dos = sec_scc.m_create(section_atom_projected_dos)
                    else:
                        sec_dos = sec_scc.m_create(section_species_projected_dos)

                    n_l = len(dos[1:]) - 1
                    values = {
                        'm_kind': 'integrated', 'energies': energies,
                        'values_total': dos[0].to('1/joule').magnitude,
                        'values_lm': dos[1:].to('1/joule').magnitude,
                        'lm': np.column_stack((np.arange(n_l), np.zeros(n_l, dtype=np.int)))}
                    for key, val in values.items():
                        setattr(sec_dos, '%s_projected_dos_%s' % (projection_type, key), val)

                    if projection_type == 'species':
                        sec_dos.species_projected_dos_species_label = species

        def get_eigenvalues(iteration):
            data = iteration.get('eigenvalues', None)
            if data is None:
                return

            n_spin = self.out_parser.get_number_of_spin_channels()

            kpts = [data[i][0] for i in range(len(data)) if i % n_spin == 0]
            kpts = None if kpts[0] is None else kpts

            eigs, occs = [], []
            for ns in range(n_spin):
                occs.append([
                    data[i][1] for i in range(len(data)) if i % n_spin == ns])
                eigs.append([
                    data[i][2] for i in range(len(data)) if i % n_spin == ns])

            return kpts, pint.Quantity(eigs, 'hartree'), occs

        def parse_scf(iteration):
            sec_scc = sec_run.section_single_configuration_calculation[-1]
            sec_scf = sec_scc.m_create(section_scf_iteration)
            energies = iteration.get('energy_components', {})
            for key, val in energies.items():
                metainfo_key = self._energy_map.get(key, None)
                if metainfo_key is not None:
                    setattr(sec_scf, '%s_scf_iteration' % metainfo_key, val)

            scf_quantities = {
                'humo': 'energy_reference_highest_occupied_iteration',
                'lumo': 'energy_reference_lowest_unoccupied_iteration',
                'fermi_level': 'energy_reference_fermi_iteration'
            }
            for key, quantity in scf_quantities.items():
                val = iteration.get(key)
                if val is not None:
                    unit = val.units
                    val = [val.magnitude] * self.out_parser.get_number_of_spin_channels()
                    setattr(sec_scf, quantity, pint.Quantity(val, unit))

            # eigenvalues are properties of
            eigenvalues = get_eigenvalues(iteration)
            if eigenvalues is not None:
                sec_eigenvalues = sec_scc.m_create(section_eigenvalues)
                if eigenvalues[0] is not None:
                    sec_eigenvalues.eigenvalues_kpoints = eigenvalues[0]
                sec_eigenvalues.eigenvalues_values = eigenvalues[1]
                sec_eigenvalues.eigenvalues_occupation = eigenvalues[2]

        def parse_gw(section):
            sec_scc = sec_run.section_single_configuration_calculation[-1]
            sec_scf_iteration = sec_scc.section_scf_iteration[-1]
            gw_scf_energies = section.get('gw_self_consistency', [])
            for energies in gw_scf_energies:
                sec_scf_iteration = sec_scc.m_create(section_scf_iteration)
                for key, val in energies.items():
                    metainfo_key = self._energy_map.get(key, None)
                    if metainfo_key is not None:
                        setattr(sec_scf_iteration, metainfo_key, val)

            self._electronic_structure_method = 'scGW' if len(gw_scf_energies) > 1 else 'G0W0'
            gw_eigenvalues = section.get('gw_eigenvalues')
            if gw_eigenvalues is None:
                return
            # TODO old parser cannot handle arrays, refactor metainfo for gw eigenvalues
            xsec_eig_group = sec_scf_iteration.m_create(x_fhi_aims_section_eigenvalues_group)
            xsec_eig_spin = xsec_eig_group.m_create(x_fhi_aims_section_eigenvalues_spin)
            xsec_eig_list = xsec_eig_spin.m_create(x_fhi_aims_section_eigenvalues_list_perturbativeGW)
            xsec_eig_list.x_fhi_aims_eigenvalue_occupation_perturbativeGW = gw_eigenvalues['occ_num']
            xsec_eig_list.x_fhi_aims_eigenvalue_ks_GroundState = gw_eigenvalues['e_gs']
            xsec_eig_list.x_fhi_aims_eigenvalue_ExactExchange_perturbativeGW = gw_eigenvalues['e_x^ex']
            xsec_eig_list.x_fhi_aims_eigenvalue_ks_ExchangeCorrelation = gw_eigenvalues['e_xc^gs']
            xsec_eig_list.x_fhi_aims_eigenvalue_correlation_perturbativeGW = gw_eigenvalues['e_c^nloc']
            xsec_eig_list.x_fhi_aims_eigenvalue_quasiParticle_energy = gw_eigenvalues['e_qp']

            sec_calc_to_calc_refs = sec_run.m_create(section_calculation_to_calculation_refs)
            sec_calc_to_calc_refs.calculation_to_calculation_ref = sec_scc
            sec_calc_to_calc_refs.calculation_to_calculation_kind = 'pertubative GW'

        def parse_vdW(section):
            # these are not actually vdW outputs but vdW control parameters but are
            # printed within the calculation section.
            # TODO why is x_fhi_aims_section_vdW_TS under x_fhi_aims_section_controlInOut_atom_species
            # we would then have to split the vdW parameters by species
            atoms = section.get('vdW_TS', {}).get('atom_hirshfeld', [])
            # get species from section_atom_type
            sec_atom_type = sec_run.section_topology[-1].section_atom_type
            if not sec_atom_type:
                return
            sec_atom_type = sec_atom_type[-1]
            sec_atom_species = sec_atom_type.x_fhi_aims_section_controlInOut_atom_species
            property_map = {
                'atom': 'x_fhi_aims_atom_type_vdW',
                'Free atom volume': 'x_fhi_aims_free_atom_volume',
                'Hirshfeld charge': 'x_fhi_aims_hirschfeld_charge',
                'Hirshfeld volume': 'x_fhi_aims_hirschfeld_volume'}
            for sec in sec_atom_species:
                for atom in atoms:
                    if sec.x_fhi_aims_controlInOut_species_name == atom['atom']:
                        sec_vdW_ts = sec.m_create(x_fhi_aims_section_vdW_TS)
                        for key, val in atom.items():
                            metainfo_name = property_map.get(key, None)
                            if metainfo_name is None:
                                continue
                            setattr(sec_vdW_ts, metainfo_name, val)
                            # TODO add the remanining properties
            self._electronic_structure_method = 'DFT'
            sec_run.section_method[-1].van_der_Waals_method = 'TS'

        def parse_section(section):
            lattice_vectors = section.get(
                'lattice_vectors', self.out_parser.get('lattice_vectors'))
            labels_positions = section.get(
                'labels_positions', self.out_parser.get('labels_positions'))
            pbc = [lattice_vectors is not None] * 3
            if labels_positions is None:
                return

            sec_system = sec_run.m_create(section_system)
            if lattice_vectors is not None:
                sec_system.lattice_vectors = lattice_vectors
                sec_system.simulation_cell = lattice_vectors

            sec_system.configuration_periodic_dimensions = pbc
            sec_system.atom_labels = labels_positions[0]
            sec_system.atom_positions = pint.Quantity(labels_positions[1], 'angstrom')
            if len(labels_positions) > 2:
                sec_system.atom_velocities = pint.Quantity(labels_positions[2], 'angstrom/ps')

            sec_scc = sec_run.m_create(section_single_configuration_calculation)
            sec_scc.single_configuration_calculation_to_system_ref = sec_system
            sec_scc.single_configuration_to_calculation_method_ref = sec_run.section_method[-1]
            energy = section.get('energy', {})
            energy.update(section.get('energy_components', [{}])[-1])
            energy.update(section.get('energy_xc', {}))
            for key, val in energy.items():
                metainfo_key = self._energy_map.get(key, None)
                if key == 'vdW energy correction':
                    sec_energy_vdw = sec_scc.m_create(section_energy_van_der_Waals)
                    kind = section.get('vdW_TS', {}).get('kind', 'Tkatchenko/Scheffler 2009')
                    sec_energy_vdw.energy_van_der_Waals_kind = kind
                    sec_energy_vdw.energy_van_der_Waals = val
                elif metainfo_key is not None:
                    setattr(sec_scc, metainfo_key, val)

            # TODO add force contributions and stress
            forces = section.get('forces', None)
            if forces is not None:
                sec_scc.atom_forces_free = forces
            time_calculation = section.get('time_force_evaluation')
            if time_calculation is not None:
                sec_scc.time_calculation = time_calculation

            scf_iterations = section.get('self_consistency')
            if scf_iterations is None:
                return
            sec_scc.number_of_scf_iterations = len(scf_iterations)
            for scf_iteration in scf_iterations:
                parse_scf(scf_iteration)

            sec_scc.single_configuration_calculation_converged = section.get(
                'converged') == 'converged'

            # density of states
            parse_dos(section)

            # fermi level
            fermi_energy = scf_iterations[-1].get('fermi_level')
            fermi_energy = fermi_energy.to('joule').magnitude if fermi_energy else 0.0
            sec_scc.energy_reference_fermi = [
                fermi_energy] * self.out_parser.get_number_of_spin_channels()

            # gw
            parse_gw(section)

            # vdW parameters
            parse_vdW(section)

        for section in self.out_parser.get('full_scf', []):
            parse_section(section)

        for section in self.out_parser.get('geometry_optimization', []):
            parse_section(section)

        for section in self.out_parser.get('molecular_dynamics', []):
            parse_section(section)

        if not sec_run.section_single_configuration_calculation:
            return

        # bandstructure
        parse_bandstructure()

        # sampling method
        sec_sampling_method = sec_run.m_create(section_sampling_method)
        sec_sampling_method.sampling_method = self.out_parser.get_calculation_type()

        # frame_sequence
        sec_frame_sequence = sec_run.m_create(section_frame_sequence)
        sec_frame_sequence.frame_sequence_to_sampling_ref = sec_sampling_method
        sec_frame_sequence.frame_sequence_local_frames_ref = sec_run.section_single_configuration_calculation

        if self._electronic_structure_method in ['DFT', 'G0W0', 'scGW']:
            sec_method = sec_run.section_method[-1]
            sec_method.electronic_structure_method = self._electronic_structure_method
            kind = 'core_settings' if self._electronic_structure_method == 'DFT' else 'starting_point'
            sec_method_to_method_refs = sec_method.m_create(section_method_to_method_refs)
            sec_method_to_method_refs.method_to_method_kind = kind
            sec_method_to_method_refs.method_to_method_ref = sec_run.section_method[-1]

    def parse_method(self):
        sec_run = self.archive.section_run[-1]
        sec_method = sec_run.m_create(section_method)

        # control parameters from out file
        self.control_parser.mainfile = self.filepath
        # we use species as marker that control parameters are printed in out file
        species = self.control_parser.get('species')
        # if not in outfile read it from control.in
        if species is None:
            control_file = self.get_fhiaims_file('control.in')
            if not control_file:
                control_file = [os.path.join(self.out_parser.maindir, 'control.in')]
            self.control_parser.mainfile = control_file[0]

        def parse_basis_set(species):
            sec_basis_set = sec_method.m_create(x_fhi_aims_section_controlIn_basis_set)
            basis_funcs = ['gaussian', 'hydro', 'valence', 'ion_occ', 'ionic', 'confined']
            for key, val in species.items():
                if key == 'species':
                    sec_basis_set.x_fhi_aims_controlIn_species_name = val[0]
                elif key == 'angular_grids':
                    sec_basis_set.x_fhi_aims_controlIn_angular_grids_method = val[0]
                elif key == 'division':
                    pass
                elif key in basis_funcs:
                    sec_basis_func = sec_basis_set.m_create(
                        x_fhi_aims_section_controlIn_basis_func)
                    sec_basis_func.x_fhi_aims_controlIn_basis_func_type = key
                    sec_basis_func.x_fhi_aims_controlIn_basis_func_n = int(val[0][0])
                    sec_basis_func.x_fhi_aims_controlIn_basis_func_l = str(val[0][1])
                    if len(val[0]) == 4 and isinstance(val[0][3], float):
                        sec_basis_func.x_fhi_aims_controlIn_basis_func_radius = val[0][3]
                else:
                    setattr(sec_basis_set, 'x_fhi_aims_controlIn_%s' % key, val[0])

            # is the number of basis functions equal to number of divisions?
            division = species.get('division', None)
            if division is not None:
                sec_basis_set.x_fhi_aims_controlIn_number_of_basis_func = len(division)
                sec_basis_set.x_fhi_aims_controlIn_division = division

        for key, val in self.control_parser.items():
            if val is None or (isinstance(val, str) and val == 'none'):
                continue
            if key.startswith('x_fhi_aims_controlIn'):
                setattr(sec_method, key, val)
            elif key == 'occupation_type':
                sec_method.x_fhi_aims_controlIn_occupation_type = val[0]
                sec_method.x_fhi_aims_controlIn_occupation_width = val[1]
                if len(val) > 2:
                    sec_method.x_fhi_aims_controlIn_occupation_order = int(val[2])
            elif key == 'relativistic':
                sec_method.x_fhi_aims_controlIn_relativistic = ' '.join(val[:2])
                if len(val) > 2:
                    sec_method.x_fhi_aims_controlIn_relativistic_threshold = val[2]
            elif key == 'species':
                for species in val:
                    parse_basis_set(species)
            elif key == 'xc':
                if isinstance(val, str):
                    val = [val]
                xc = ' '.join([v for v in val if isinstance(v, str)])
                sec_method.x_fhi_aims_controlIn_xc = str(xc)
                if not isinstance(val[-1], str) and xc.lower().startswith('hse'):
                    unit = self.control_parser.get('x_fhi_aims_controlIn_hse_unit')
                    hse_omega = pint.Quantity(val[-1], unit) if unit else val[-1]
                    sec_method.x_fhi_aims_controlIn_hse_omega = hse_omega
                hybrid_coeff = self.control_parser.get('x_fhi_aims_controlIn_hybrid_xc_coeff')
                if hybrid_coeff is not None:
                    # is it necessary to check if xc is a hybrid type aside from hybrid_coeff
                    sec_method.x_fhi_aims_controlIn_hybrid_xc_coeff = hybrid_coeff

        # atom species
        self.parse_topology()

        # xc functional from output
        xc_inout = self.out_parser.get('x_fhi_aims_controlInOut_xc', None)
        if xc_inout is not None:
            xc_inout = [xc_inout] if isinstance(xc_inout, str) else xc_inout
            xc = ' '.join([v for v in xc_inout if isinstance(v, str)])
            sec_method.x_fhi_aims_controlInOut_xc = str(xc)

            # hse func
            hse_omega = None
            if not isinstance(xc_inout[-1], str) and xc.lower().startswith('hse'):
                unit = self.out_parser.get('x_fhi_aims_controlInOut_hse_unit')
                hse_omega = pint.Quantity(xc_inout[-1], unit) if unit else xc_inout[-1]
                sec_method.x_fhi_aims_controlInOut_hse_omega = hse_omega

            hybrid_coeff = None
            hybrid_coeff = self.out_parser.get('x_fhi_aims_controlInOut_hybrid_xc_coeff')
            if hybrid_coeff is not None:
                sec_method.x_fhi_aims_controlIn_hybrid_xc_coeff = hybrid_coeff

            # convert parsed xc to meta info
            xc_meta_list = self._xc_map.get(xc, [])
            for xc_meta in xc_meta_list:
                sec_xc_func = sec_method.m_create(section_XC_functionals)
                sec_xc_func.XC_functional_name = xc_meta.get('name')
                xc_parameters = dict()
                if hse_omega is not None:
                    xc_parameters.setdefault('$\\omega$ in m^-1', hse_omega)
                if hybrid_coeff is not None:
                    xc_parameters.setdefault('hybrid coefficient $\\alpha$', hybrid_coeff)
                    weight = xc_meta.get('weight', None)
                    if weight is not None:
                        sec_xc_func.XC_functional_weight = weight(hybrid_coeff)
                if xc_parameters:
                    sec_xc_func.XC_functional_parameters = xc_parameters

    def parse_topology(self):
        sec_run = self.archive.section_run[-1]
        sec_topology = sec_run.m_create(section_topology)

        def parse_atom_type(species):
            sec_atom_type = sec_topology.m_create(section_atom_type)
            sec_atom_species = sec_atom_type.m_create(
                x_fhi_aims_section_controlInOut_atom_species)
            for key, val in species.items():
                if key == 'nuclear charge':
                    charge = pint.Quantity(val[0], 'elementary_charge')
                    sec_atom_type.atom_type_charge = charge
                    sec_atom_species.x_fhi_aims_controlInOut_species_charge = charge
                elif key == 'atomic mass':
                    mass = pint.Quantity(val[0][0], val[0][1])
                    sec_atom_type.atom_type_mass = mass
                    sec_atom_species.x_fhi_aims_controlInOut_species_mass = mass
                elif key == 'species':
                    sec_atom_type.atom_type_name = val[0]
                    sec_atom_species.x_fhi_aims_controlInOut_species_name = val[0]
                elif 'request to include pure gaussian fns' in key:
                    sec_atom_species.x_fhi_aims_controlInOut_pure_gaussian = val[0]
                elif 'cutoff potl' in key:
                    sec_atom_species.x_fhi_aims_controlInOut_species_cut_pot = pint.Quantity(
                        val[0][0], 'angstrom')
                    sec_atom_species.x_fhi_aims_controlInOut_species_cut_pot_width = pint.Quantity(
                        val[0][1], 'angstrom')
                    sec_atom_species.x_fhi_aims_controlInOut_species_cut_pot_scale = val[0][2]
                elif 'free-atom valence' in key:
                    for i in range(len(val)):
                        sec_basis_func = sec_atom_species.m_create(
                            x_fhi_aims_section_controlInOut_basis_func)
                        sec_basis_func.x_fhi_aims_controlInOut_basis_func_type = 'free-atom valence'
                        sec_basis_func.x_fhi_aims_controlInOut_basis_func_n = val[i][0]
                        sec_basis_func.x_fhi_aims_controlInOut_basis_func_l = val[i][1]
                        sec_basis_func.x_fhi_aims_controlInOut_basis_func_occ = val[i][2]
                elif 'hydrogenic basis' in key:
                    for i in range(len(val)):
                        sec_basis_func = sec_atom_species.m_create(
                            x_fhi_aims_section_controlInOut_basis_func)
                        sec_basis_func.x_fhi_aims_controlInOut_basis_func_type = 'hydrogenic basis'
                        sec_basis_func.x_fhi_aims_controlInOut_basis_func_n = val[i][0]
                        sec_basis_func.x_fhi_aims_controlInOut_basis_func_l = val[i][1]
                        sec_basis_func.x_fhi_aims_controlInOut_basis_func_eff_charge = val[i][2]
                elif 'ionic basis' in key:
                    for i in range(len(val)):
                        sec_basis_func = sec_atom_species.m_create(
                            x_fhi_aims_section_controlInOut_basis_func)
                        sec_basis_func.x_fhi_aims_controlInOut_basis_func_type = 'ionic basis'
                        sec_basis_func.x_fhi_aims_controlInOut_basis_func_n = val[i][0]
                        sec_basis_func.x_fhi_aims_controlInOut_basis_func_l = val[i][1]
                elif 'basis function' in key:
                    for i in range(len(val)):
                        sec_basis_func = sec_atom_species.m_create(
                            x_fhi_aims_section_controlInOut_basis_func)
                        sec_basis_func.x_fhi_aims_controlInOut_basis_func_type = key.split(
                            'basis')[0].strip()
                        if val[i][0] == 'L':
                            sec_basis_func.x_fhi_aims_controlInOut_basis_func_gauss_l = val[i][2]
                            sec_basis_func.x_fhi_aims_controlInOut_basis_func_gauss_N = val[i][3]
                            alpha = [val[i][j + 2] for j in range(len(val[i])) if val[i][j] == 'alpha']
                            weight = [val[i][j + 2] for j in range(len(val[i])) if val[i][j] == 'weight']
                            alpha = pint.Quantity(alpha, '1/angstrom ** 2')
                            sec_basis_func.x_fhi_aims_controlInOut_basis_func_gauss_alpha = alpha
                            sec_basis_func.x_fhi_aims_controlInOut_basis_func_gauss_weight = weight
                        elif len(val[i]) == 2:
                            sec_basis_func.x_fhi_aims_controlInOut_basis_func_gauss_l = val[i][0]
                            alpha = pint.Quantity(val[i][1], '1/angstrom ** 2')
                            sec_basis_func.x_fhi_aims_controlInOut_basis_func_primitive_gauss_alpha = alpha

        # add inout parameters read from main output
        # species
        species = self.out_parser['control_inout']['species']
        if species is not None:
            for specie in species:
                parse_atom_type(specie)

    def _init_parsers(self):
        self.out_parser.mainfile = self.filepath
        self.out_parser.logger = self.logger
        self.control_parser.logger = self.logger
        self.dos_parser.logger = self.logger
        self.bandstructure_parser.logger = self.logger

    def parse(self, filepath, archive, logger):
        self.filepath = os.path.abspath(filepath)
        self.archive = archive
        self.maindir = os.path.dirname(self.filepath)
        self.logger = logger if logger is not None else logging

        self._electronic_structure_method = 'DFT'
        self._init_parsers()

        sec_run = self.archive.m_create(section_run)
        sec_run.program_name = 'FHI-aims'
        sec_run.program_basis_set_type = 'numeric AOs'
        section_run_keys = [
            'program_version', 'x_fhi_aims_program_compilation_date',
            'x_fhi_aims_program_compilation_time', 'program_compilation_host',
            'x_fhi_aims_program_execution_date', 'x_fhi_aims_program_execution_time',
            'time_run_cpu1_start', 'time_run_wall_start', 'raw_id',
            'x_fhi_aims_number_of_tasks']
        for key in section_run_keys:
            value = self.out_parser.get(key)
            if value is None:
                continue
            setattr(sec_run, key, value)

        sec_parallel_tasks = sec_run.m_create(x_fhi_aims_section_parallel_tasks)
        # why embed section not just let task be an array

        task_nrs = self.out_parser.get('x_fhi_aims_parallel_task_nr')
        task_hosts = self.out_parser.get('x_fhi_aims_parallel_task_host')
        for i in range(len(task_nrs)):
            sec_parallel_task_assignement = sec_parallel_tasks.m_create(
                x_fhi_aims_section_parallel_task_assignement)
            sec_parallel_task_assignement.x_fhi_aims_parallel_task_nr = task_nrs[i]
            sec_parallel_task_assignement.x_fhi_aims_parallel_task_host = task_hosts[i]

        self.parse_method()

        self.parse_configurations()
