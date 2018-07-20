# Copyright 2015-2018 Franz Knuth, Fawzi Mohamed, Wael Chibani, Ankit Kariryaa, Lauri Himanen, Danio Brambila
# 
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from __future__ import absolute_import
try:
    import setup_paths
except ImportError:
    pass
import numpy as np
from nomadcore.local_meta_info import loadJsonFile, InfoKindEl
from nomadcore.unit_conversion.unit_conversion import convert_unit
import json, os, re

############################################################
# This file contains functions that are needed
# by more than one parser.
############################################################

def get_metaInfo(filePath):
    """Loads metadata.

    Args:
        filePath: Location of metadata.

    Returns:
        metadata which is an object of the class InfoKindEnv in nomadcore.local_meta_info.py.
    """
    metaInfoEnv, warnings = loadJsonFile(filePath = filePath, dependencyLoader = None, extraArgsHandling = InfoKindEl.ADD_EXTRA_ARGS, uri = None)
    return metaInfoEnv

def write_controlIn(backend, metaInfoEnv, valuesDict, writeXC, location, logger):
    """Writes the values of the control.in file.

    Write the last occurrence of a keyword, i.e. [-1], since aims uses the last occurrence of a keyword.

    ATTENTION
    backend.superBackend is used here instead of only the backend to write the JSON values,
    since this allows to bybass the caching setting which was used to collect the values for processing.
    However, this also bypasses the checking of validity of the metadata name by the backend.
    The scala part will check the validity nevertheless.

    Args:
        backend: Class that takes care of writing and caching of metadata.
        metaInfoEnv: Loaded metadata.
        valuesDict: Dictionary that contains the cached values of a section.
        writeXC: Boolean that determines whether the keywords related to the xc functional should be written.
        location: A string that is used to specify where more than one setting for xc was found.
        logger: Logging object where messages should be written to.
    """
    # list of excluded metadata for writeout
    # k_grid is written with k1 so that k2 and k2 are not needed.
    # fhi_aims_controlIn_relativistic_threshold is written with fhi_aims_controlIn_relativistic.
    # The xc setting have to be handeled separatly since having more than one gives undefined behavior.
    # hse_omega is only written if HSE was used and converted according to hse_unit which is not written since not needed.
    # hybrid_xc_coeff is only written for hybrid functionals.
    # verbatim_writeout is only needed to detect if the control.in is written in the main file of aims.
    exclude_list = [
                    'x_fhi_aims_controlIn_k2',
                    'x_fhi_aims_controlIn_k3',
                    'x_fhi_aims_controlIn_relativistic_threshold',
                    'x_fhi_aims_controlIn_xc',
                    'x_fhi_aims_controlIn_hse_omega',
                    'x_fhi_aims_controlIn_hse_unit',
                    'x_fhi_aims_controlIn_hybrid_xc_coeff',
                    'x_fhi_aims_controlIn_verbatim_writeout',
                   ]
    # write settings
    for k,v in valuesDict.items():
        if k.startswith('x_fhi_aims_controlIn_'):
            if k in exclude_list:
                continue
            # write k_krid
            elif k == 'x_fhi_aims_controlIn_k1':
                write_k_grid(backend, 'x_fhi_aims_controlIn_k', valuesDict)
            elif k == 'x_fhi_aims_controlIn_relativistic':
                # check for scalar ZORA setting and convert to one common name
                if re.match(r"\s*zora\s+scalar", v[-1], re.IGNORECASE):
                    backend.superBackend.addValue(k, 'zora scalar')
                    # write threshold only for scalar ZORA
                    value = valuesDict[k + '_threshold']
                    if value is not None:
                        backend.superBackend.addValue(k + '_threshold', value[-1])
                else:
                    backend.superBackend.addValue(k, v[-1].lower())
            # default writeout
            else:
                # convert keyword values of control.in which are strings to lowercase for consistency
                if isinstance(v[-1], str):
                    value = v[-1].lower()
                else:
                    value = v[-1]
                backend.superBackend.addValue(k, value)
    # handling of xc functional
    if writeXC:
        write_xc_functional(backend = backend,
            metaInfoEnv = metaInfoEnv,
            metaNameStart = 'x_fhi_aims_controlIn',
            valuesDict = valuesDict,
            location = 'control.in',
            logger = logger)

def write_k_grid(backend, metaName, valuesDict):
    """Function to write k-grid for controlIn and controlInOut.

    Args:
        backend: Class that takes care of writing and caching of metadata.
        metaName: Corresponding metadata name for k.
        valuesDict: Dictionary that contains the cached values of a section.
    """
    k_grid = []
    for i in ['1', '2', '3']:
        ki = valuesDict.get(metaName + i)
        if ki is not None:
            k_grid.append(ki[-1])
    if k_grid:
        backend.superBackend.addArrayValues(metaName + '_grid', np.asarray(k_grid))

def write_xc_functional(backend, metaInfoEnv, metaNameStart, valuesDict, location, logger):
    """Function to write xc settings for controlIn and controlInOut.

    The omega of the HSE-functional is converted to the unit given in the metadata.
    The xc functional from controlInOut is converted and writen in the format specified in the metadata.

    Args:
        backend: Class that takes care of writing and caching of metadata.
        metaInfoEnv: Loaded metadata.
        metaNameStart: Base name of metdata for xc. Must be fhi_aims_controlIn or fhi_aims_controlInOut.
        valuesDict: Dictionary that contains the cached values of a section.
        location: A string that is used to specify where more than one setting for xc was found.
        logger: Logging object where messages should be written to.
    """
    # two functions to convert hybrid_xc_coeff to the correct weight
    def GGA_weight(x):
        return 1.0 - x
    def HF_weight(x):
        return x
    # TODO vdW functionals, double-hybrid functionals, screx, and cohsex
    # Dictionary for conversion of xc functional name in aims to metadata format.
    # The individual x and c components of the functional are given as dictionaries.
    # Possible keys of such a dictionary are 'name', 'weight', and 'convert'.
    # If 'weight' is not given it is not written.
    # With 'convert', a funtion is specified how hybrid_xc_coeff is converted to the correct weight for this xc component.
    xcDict = {
              'Perdew-Wang parametrisation of Ceperley-Alder LDA':   [{'name': 'LDA_C_PW'}, {'name': 'LDA_X'}],
              'Perdew-Zunger parametrisation of Ceperley-Alder LDA': [{'name': 'LDA_C_PZ'}, {'name': 'LDA_X'}],
              'VWN-LDA parametrisation of VWN5 form':                [{'name': 'LDA_C_VWN'}, {'name': 'LDA_X'}],
              'VWN-LDA parametrisation of VWN-RPA form':             [{'name': 'LDA_C_VWN_RPA'}, {'name': 'LDA_X'}],
              'AM05 gradient-corrected functionals':                 [{'name': 'GGA_C_AM05'}, {'name': 'GGA_X_AM05'}],
              'BLYP functional':                                     [{'name': 'GGA_C_LYP'}, {'name': 'GGA_X_B88'}],
              'PBE gradient-corrected functionals':                  [{'name': 'GGA_C_PBE'}, {'name': 'GGA_X_PBE'}],
              'PBEint gradient-corrected functional':                [{'name': 'GGA_C_PBEINT'}, {'name': 'GGA_X_PBEINT'}],
              'PBEsol gradient-corrected functionals':               [{'name': 'GGA_C_PBE_SOL'}, {'name': 'GGA_X_PBE_SOL'}],
              'RPBE gradient-corrected functionals':                 [{'name': 'GGA_C_PBE'}, {'name': 'GGA_X_RPBE'}],
              'revPBE gradient-corrected functionals':               [{'name': 'GGA_C_PBE'}, {'name': 'GGA_X_PBE_R'}],
              'PW91 gradient-corrected functionals':                 [{'name': 'GGA_C_PW91'}, {'name': 'GGA_X_PW91'}],
              'M06-L gradient-corrected functionals':                [{'name': 'MGGA_C_M06_L'}, {'name': 'MGGA_X_M06_L'}],
              'M11-L gradient-corrected functionals':                [{'name': 'MGGA_C_M11_L'}, {'name': 'MGGA_X_M11_L'}],
              'TPSS gradient-corrected functionals':                 [{'name': 'MGGA_C_TPSS'}, {'name': 'MGGA_X_TPSS'}],
              'TPSSloc gradient-corrected functionals':              [{'name': 'MGGA_C_TPSSLOC'}, {'name': 'MGGA_X_TPSS'}],
              'hybrid B3LYP functional':                             [{'name': 'HYB_GGA_XC_B3LYP5'}],
              'Hartree-Fock':                                        [{'name': 'HF_X'}],
              'HSE':                                                 [{'name': 'HYB_GGA_XC_HSE03'}],
              'HSE-functional':                                      [{'name': 'HYB_GGA_XC_HSE06'}],
              'hybrid-PBE0 functionals':                             [{'name': 'GGA_C_PBE'}, {'name': 'GGA_X_PBE', 'weight': 0.75, 'convert': GGA_weight}, {'name': 'HF_X', 'weight': 0.25, 'convert': HF_weight}],
              'hybrid-PBEsol0 functionals':                          [{'name': 'GGA_C_PBE_SOL'}, {'name': 'GGA_X_PBE_SOL', 'weight': 0.75, 'convert': GGA_weight}, {'name': 'HF_X', 'weight': 0.25, 'convert': HF_weight}],
              'Hybrid M06 gradient-corrected functionals':           [{'name': 'HYB_MGGA_XC_M06'}],
              'Hybrid M06-2X gradient-corrected functionals':        [{'name': 'HYB_MGGA_XC_M06_2X'}],
              'Hybrid M06-HF gradient-corrected functionals':        [{'name': 'HYB_MGGA_XC_M06_HF'}],
              'Hybrid M08-HX gradient-corrected functionals':        [{'name': 'HYB_MGGA_XC_M08_HX'}],
              'Hybrid M08-SO gradient-corrected functionals':        [{'name': 'HYB_MGGA_XC_M08_SO'}],
              'Hybrid M11 gradient-corrected functionals':           [{'name': 'HYB_MGGA_XC_M11'}],
             }
    # description for hybrid coefficient
    xcHybridCoeffDescr = 'hybrid coefficient $\\alpha$'
    # descritpion for omega of HSE06
    xcOmegaDescr = '$\\omega$ in m^-1'
    # distinguish between control.in and the aims output from the parsed control.in
    if metaNameStart == 'x_fhi_aims_controlIn':
        hseFunc = 'hse06'
        # functionals where hybrid_xc_coeff is written
        writeHybridCoeff = ['b3lyp', 'hse03', 'hse06', 'pbe0', 'pbesol0']
        xcWrite = False
    elif metaNameStart == 'x_fhi_aims_controlInOut':
        hseFunc = 'HSE-functional'
        # functionals where hybrid_xc_coeff is written
        writeHybridCoeff = ['hybrid B3LYP functional', 'HSE', 'HSE-functional', 'hybrid-PBE0 functionals', 'hybrid-PBEsol0 functionals']
        xcWrite = True
    else:
        logger.error("Unknown metaNameStart %s in function write_xc_functional in %s. Please correct." % (metaNameStart, os.path.basename(__file__)))
        return
    # get cached values for xc functional
    xc = valuesDict.get(metaNameStart + '_xc')
    if xc is not None:
        # check if only one xc keyword was found in control.in
        if len(xc) > 1:
            logger.error("Found %d settings for the xc functional in %s: %s. This leads to an undefined behavior of the calculation and no metadata can be written for %s_xc." % (len(xc), location, xc, metaNameStart))
        else:
            backend.superBackend.addValue(metaNameStart + '_xc', xc[-1])
            # check for hybrid_xc_coeff
            hybridCoeff = valuesDict.get(metaNameStart + '_hybrid_xc_coeff')
            # write hybrid_xc_coeff for certain functionals
            if hybridCoeff is not None and xc[-1] in writeHybridCoeff:
                backend.superBackend.addValue(metaNameStart + '_hybrid_xc_coeff', hybridCoeff[-1])
            # hse_omega is only written for HSE06
            if xc[-1] == hseFunc:
                hse_omega = valuesDict.get(metaNameStart + '_hse_omega')
                if hse_omega is not None:
                    # get unit from metadata
                    unit = metaInfoEnv.infoKinds[metaNameStart + '_hse_omega'].units
                    # try unit conversion and write hse_omega
                    hse_unit = valuesDict.get(metaNameStart + '_hse_unit')
                    if hse_unit is not None:
                        omegaValue = None
                        if hse_unit[-1] in ['a', 'A']:
                            omegaValue = convert_unit(hse_omega[-1], 'angstrom**-1', unit)
                        elif hse_unit[-1] in ['b', 'B']:
                            omegaValue = convert_unit(hse_omega[-1], 'bohr**-1', unit)
                        if omegaValue is not None:
                            backend.superBackend.addValue(metaNameStart + '_hse_omega', omegaValue)
                        else:
                            logger.warning("Unknown hse_unit %s in %s. Cannot write %s" % (hse_unit[-1], location, metaNameStart + '_hse_omega'))
                    else:
                        logger.warning("No value found for %s. Cannot write %s" % (metaNameStart + '_hse_unit', metaNameStart + '_hse_omega'))
            # convert xc functional for metadata
            if xcWrite:
                # get list of xc components according to parsed value
                xcList = xcDict.get(xc[-1])
                if xcList is not None:
                    # loop over the xc components
                    for xcItem in xcList:
                        xcName = xcItem.get('name')
                        if xcName is not None:
                            # write section and and XC_functional_name
                            gIndexTmp = backend.openSection('section_XC_functionals')
                            backend.addValue('XC_functional_name', xcName)
                            # write hybrid_xc_coeff for B3LYP and HSE03 into XC_functional_parameters
                            if hybridCoeff is not None and xc[-1] in ['hybrid B3LYP functional', 'HSE']:
                                backend.addValue('XC_functional_parameters', {xcHybridCoeffDescr: hybridCoeff[-1]})
                            # write omega and hybrid_xc_coeff for HSE06
                            elif xc[-1] == hseFunc:
                                # converted value of omega was obtained above
                                parameters = {xcOmegaDescr: omegaValue}
                                # add hybrid_xc_coeff
                                if hybridCoeff is not None:
                                    hybrid = hybridCoeff[-1]
                                else:
                                    hybrid = 0.25
                                parameters[xcHybridCoeffDescr] = hybrid
                                backend.addValue('XC_functional_parameters', parameters)
                            # adjust weight of functionals that are affected by hybrid_xc_coeff
                            elif hybridCoeff is not None and 'convert' in xcItem:
                                backend.addValue('XC_functional_weight', xcItem['convert'](hybridCoeff[-1]))
                            # write weight if present for current xcItem
                            else:
                                xcWeight = xcItem.get('weight')
                                if xcWeight is not None:
                                    backend.addValue('XC_functional_weight', xcWeight)
                            backend.closeSection('section_XC_functionals', gIndexTmp)
                        else:
                            logger.error("The dictionary for xc functional '%s' does not have the key 'name'. Please correct the dictionary xcDict in %s." % (xc[-1], os.path.basename(__file__)))
                else:
                    logger.error("The xc functional '%s' could not be converted for the metadata. Please add it to the dictionary xcDict in %s." % (xc[-1], os.path.basename(__file__)))
