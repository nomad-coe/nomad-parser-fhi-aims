import setup_paths
import numpy as np
from nomadcore.unit_conversion.unit_conversion import convert_unit
import os

############################################################
# This file contains functions that are needed
# by more than one parser.
############################################################

def write_k_grid(backend, metaName, valuesDict):
    """Function to write k-grid for controlIn and controlInOut.

    Args:
        backend: Takes care of the writting of the metadata.
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
    The xc functional from controlInOut is converted to the string required for the metadata XC_functional.

    Args:
        backend: Takes care of the writting of the metadata.
        metaInfoEnv: Loaded metadata.
        metaNameStart: Base name of metdata for xc. Must be fhi_aims_controlIn or fhi_aims_controlInOut.
        valuesDict: Dictionary that contains the cached values of a section.
        location: A string that is used to specify where more than one setting for xc was found.
        logger: Logging object where messages should be written to.
    """
    # dictionary for conversion of xc functional name in aims to metadata format
    # TODO hybrid GGA and mGGA functionals and vdW functionals
    xcDict = {
              'Perdew-Wang parametrisation of Ceperley-Alder LDA': 'LDA_C_PW_LDA_X',
              'Perdew-Zunger parametrisation of Ceperley-Alder LDA': 'LDA_C_PZ_LDA_X',
              'VWN-LDA parametrisation of VWN5 form': 'LDA_C_VWN_LDA_X',
              'VWN-LDA parametrisation of VWN-RPA form': 'LDA_C_VWN_RPA_LDA_X',
              'AM05 gradient-corrected functionals': 'GGA_C_AM05_GGA_X_AM05',
              'BLYP functional': 'GGA_C_LYP_GGA_X_B88',
              'PBE gradient-corrected functionals': 'GGA_C_PBE_GGA_X_PBE',
              'PBEint gradient-corrected functional': 'GGA_C_PBEINT_GGA_X_PBEINT',
              'PBEsol gradient-corrected functionals': 'GGA_C_PBE_SOL_GGA_X_PBE_SOL',
              'RPBE gradient-corrected functionals': 'GGA_C_PBE_GGA_X_RPBE',
              'revPBE gradient-corrected functionals': 'GGA_C_PBE_GGA_X_PBE_R',
              'PW91 gradient-corrected functionals': 'GGA_C_PW91_GGA_X_PW91',
              'M06-L gradient-corrected functionals': 'MGGA_C_M06_L_MGGA_X_M06_L',
              'M11-L gradient-corrected functionals': 'MGGA_C_M11_L_MGGA_X_M11_L',
              'TPSS gradient-corrected functionals': 'MGGA_C_TPSS_MGGA_X_TPSS',
              'TPSSloc gradient-corrected functionals': 'MGGA_C_TPSSLOC_MGGA_X_TPSS',
              'Hartree-Fock': 'HF_X',
             }
    # distinguish between control.in and the aims output from the parsed control.in
    if metaNameStart == 'fhi_aims_controlIn':
        hseFunc = 'hse06'
        xcWrite = False
    elif metaNameStart == 'fhi_aims_controlInOut':
        hseFunc = 'HSE-functional'
        xcWrite = True
    else:
        logger.error("Unknown metaNameStart %s in function write_xc_functional in %s. Please correct." % (metaNameStart, os.path.basename(__file__)))
        return
    # get cached values for xc functional
    xc = valuesDict.get(metaNameStart + '_xc')
    if xc is not None:
        # check if only one xc keyword was found in control.in
        if len(xc) > 1:
            logger.warning("Found %d settings for the xc functional in %s: %s. This leads to an undefined behavior und no metadata can be written for %s_xc." % (len(xc), location, xc, metaNameStart))
        else:
            backend.superBackend.addValue(metaNameStart + '_xc', xc[-1])
            # convert xc functional to metadata XC_functional
            if xcWrite:
                xcMetaName = 'XC_functional'
                xcString = xcDict.get(xc[-1])
                if xcString is not None:
                    backend.addValue(xcMetaName, xcString)
                else:
                    logger.error("The xc functional '%s' could not be converted to the required string for the metadata '%s'. Please add it to the dictionary xcDict in %s." % (xc[-1], xcMetaName, os.path.basename(__file__)))
            # hse_omega is only written for HSE06
            hse_omega = valuesDict.get(metaNameStart + '_hse_omega')
            if hse_omega is not None and xc[-1] == hseFunc:
                # get unit from metadata
                unit = metaInfoEnv.infoKinds[metaNameStart + '_hse_omega'].units
                # try unit conversion and write hse_omega
                hse_unit = valuesDict.get(metaNameStart + '_hse_unit')
                if hse_unit is not None:
                    value = None
                    if hse_unit[-1] in ['a', 'A']:
                        value = convert_unit(hse_omega[-1], 'angstrom**-1', unit)
                    elif hse_unit[-1] in ['b', 'B']:
                        value = convert_unit(hse_omega[-1], 'bohr**-1', unit)
                    if value is not None:
                        backend.superBackend.addValue(metaNameStart + '_hse_omega', value)
                    else:
                        logger.warning("Unknown hse_unit %s in %s. Cannot write %s" % (hse_unit[-1], location, metaNameStart + '_hse_omega'))
                else:
                    logger.warning("No value found for %s. Cannot write %s" % (metaNameStart + '_hse_unit', metaNameStart + '_hse_omega'))

