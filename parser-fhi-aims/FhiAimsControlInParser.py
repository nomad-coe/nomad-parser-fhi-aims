
# Author: Fawzi Mohamed

import numpy
import os, sys
import re
import json
import traceback

from LineParsers import *

class NoProp(object):
    "just an enum value"
    pass

class Keyword(object):
    def __init__(self, name, dtype, outputWrites=None,defaultValue=None, description=None, superKeys=["ControlInValues"],units=None, keyParser=None,subKeywords=None,repeats=False):
        self.name=name
        self.dtype=dtype
        self.repeats=repeats
        self.outputWrites=outputWrites
        if outputWrites is None:
            self.outputWrites=[]
        self.defaultValue=defaultValue
        self.superKeys=superKeys
        self.units=units
        if description is None:
            self.description="Property for key {0} of FHI-aims".format(self.name)
            self.description += " with value of kind {0}".format(self.dtype)
            self.description+="."
        else:
            self.description=description
        self.subKeywords=subKeywords
        self.keyParser=keyParser

    @staticmethod
    def propertyNameForKeywordName(name):
        if len(name)>0 and name[0].isupper():
            return "ControlIn_"+name.lower()
        else:
            return "controlIn_"+name.lower()
    @staticmethod
    def propertyOutNameForKeywordName(name):
        if len(name)>0 and name[0].isupper():
            return "OutControlIn_"+name.lower()
        else:
            return "outControlIn_"+name.lower()

    def propertyName(self):
        return self.propertyNameForKeywordName(self.name)
    def propertyOutName(self):
        return self.propertyOutNameForKeywordName(self.name)

    def property(self):
        if self.dtype is NoProp:
            return None
        prop=Property.withName(self.propertyName())
        if prop is None:
            prop=Prop(self.propertyName(),
            description=self.description,
            superKeys=self.superKeys,
            repeats=self.repeats,
            dtype=self.dtype,
            units=self.units)
        return prop

    def outProperty(self):
        if self.dtype is NoProp:
            return None
        prop=Property.withName(self.propertyOutName())
        if prop is None:
            prop=Prop(self.propertyOutName(),
            description=self.description,
            superKeys=self.superKeys,
            repeats=self.repeats,
            dtype=self.dtype,
            units=self.units)
        return prop

    def guaranteeProperties(self):
        self.property();
        self.outProperty();
        if not self.subKeywords is None:
            for k in self.subKeywords:
                k.guaranteeProperties();

    def addInputParser(self,container):
        def setDefaults(lparser):
            if (lparser.kind&MatchKind.DefaultSet)!=0:
                lparser.kind=MatchKind.Sequenced|MatchKind.AlwaysActive|MatchKind.Repeatable|MatchKind.ExpectFail
            if lparser.propNameMapper is None:
                lparser.propNameMapper=self.propertyNameForKeywordName
            for subP in lparser.subParsers:
                setDefaults(subP)
        p=None
        if self.keyParser:
            p=self.keyParser
        elif self.dtype is NoProp:
            return
        elif self.dtype is None:
            p=FixedValueParser(startReStr=r"\s*{0}\b".format(self.name),
                matchDict={ self.propertyName():True })
        else:
            startReStr=r"\s*{0}\s+(?P<{1}>\S.*)".format(self.name, self.propertyName())
            p=LineParser(self.name, startReStr=startReStr)
        setDefaults(p)
        container.append(p)
        if self.subKeywords:
            for subK in self.subKeywords:
                subK.addInputParser(p.subParsers)

    def addOutputParsers(self,container):
        def setDefaults(lparser):
            if (lparser.kind&MatchKind.DefaultSet)!=0:
                lparser.kind=MatchKind.Sequenced|MatchKind.AlwaysActive|MatchKind.Repeatable|MatchKind.ExpectFail
            if lparser.propNameMapper is None:
                lparser.propNameMapper=self.propertyOutNameForKeywordName
            for subP in lparser.subParsers:
                setDefaults(subP)
        for p in self.outputWrites:
            setDefaults(p)
            container.append(p)
            if self.subKeywords:
                for subK in self.subKeywords:
                    subK.addOutputParsers(p.subParsers)

def relax_geometryValueHandling(logger, globalContext, localContext, strVal):
    splVal=strVal.strip().split()
    geoProp=Property.withName("controlIn_relax_geometry")
    accuracyProp=Property.withName("controlIn_relax_accuracy_forces")
    if (len(splVal)):
        geoProp.handleProperty(logger,globalContext,localContext,value=splVal[0])
    if (len(splVal)>1):
        accuracyProp.handleProperty(logger,globalContext,localContext,value=accuracyProp.strToVal(splVal[1], logger))
    if (len(splVal)>2):
        logger.addWarning("Property relax_geometry has an unexpected number of arguments: {0}".format(strVal))
    return True

ctrlKeywords=[
    Keyword('xc', dtype=str,
        superKeys=["Xc-functional"],
        keyParser=SLParser(r"\s*xc\s+(?P<xc>.+)"),
        outputWrites=[
        FixedValueParser(r"\s*XC: Using Perdew-Zunger parametrisation of Ceperley-Alder LDA\.",
            matchDict={"xc":"pz-lda"}),
        FixedValueParser(r"\s*XC: Using Perdew-Wang parametrisation of Ceperley-Alder LDA\.",
            matchDict={"xc":"pw-lda"}),
        FixedValueParser(r"\s*XC: Using VWN-LDA parametrisation of VWN5 form\.",
            matchDict={"xc":"vwn"}),
        FixedValueParser(r"\s*XC: Using VWN-LDA parametrisation of VWN-RPA form\.",
            matchDict={"xc":"vwn-gauss"}),
        FixedValueParser(r"\s*XC: Using HSE-functional with OMEGA =\s*(?P<hse_omega>[-+0-9.eEdD]+)\s*<units>\.",
            matchDict={"xc":"hse06"}),
        FixedValueParser(r"\s*XC:  HSE with OMEGA_PBE =\s*[-+0-9.eEdD]+\s*bohr\^-1",
            matchDict={"xc":"hse03"}),
        FixedValueParser(r"\s*XC: Using PBE gradient-corrected functionals\.",
            matchDict={"xc":"pbe"}),
        FixedValueParser(r"\s*XC: Using PW91 gradient-corrected functionals\.",
            matchDict={"xc":"pw91_gga"}),
        FixedValueParser(r"\s*XC: Using PBE with VDW\.",
            matchDict={"xc":"pbe_vdw"}),
        FixedValueParser(r"\s*XC: Using PBEsol gradient-corrected functionals\.",
            matchDict={"xc":"pbesol"}),
        FixedValueParser(r"\s*XC: Using PBEint gradient-corrected functional\.",
            matchDict={"xc":"pbeint"}),
        FixedValueParser(r"\s*XC: Using RPBE gradient-corrected functionals\.",
            matchDict={"xc":"rpbe"}),
        FixedValueParser(r"\s*XC: Using revPBE gradient-corrected functionals\.",
            matchDict={"xc":"revpbe"}),
        FixedValueParser(r"\s*XC: Using revPBE with vdw\.",
            matchDict={"xc":"revpbe_vdw"}),
        FixedValueParser(r"\s*XC: Using AM05 gradient-corrected functionals\.",
            matchDict={"xc":"am05"}),
        FixedValueParser(r"\s*XC: Using hybrid-PBE0 functionals\.",
            matchDict={"xc":"pbe0"}),
        FixedValueParser(r"\s*Hartree calculation starts \.\.\.\.\.\.",
            matchDict={"xc":"hartree"}),
        FixedValueParser(r"\s*Hartree-Fock calculation starts \.\.\.\.\.\.",
            matchDict={"xc":"hf"}),
        FixedValueParser(r"\s*MP2 will start after the HF calculation\.",
            matchDict={"postHfMethod":"mp2"}),
        FixedValueParser(r"\s*XC: Using BLYP functional\.",
            matchDict={"xc":"blyp"}),
        FixedValueParser(r"\s*XC: Using hybrid B3LYP functional\.",
            matchDict={"xc":"b3lyp"}),
        FixedValueParser(r"\s*XC: Using hybrid-PBEsol0 functionals\.",
            matchDict={"xc":"pbesol0"}),
        FixedValueParser(r"\s*XC: Running screened exchange calculation \.\.\.",
            matchDict={"xc":"screx"}),
        FixedValueParser(r"\s*XC: Running screened exchange plus coulomb hole \(COHSEX\) calculation \.\.\.",
            matchDict={"xc":"cohsex"})
        ]),
    Keyword('dry_run', dtype=None),
    Keyword('el_ph_coupling', dtype=bool),# not implemented as of now
    Keyword('charge',dtype=float,
        outputWrites=[
        SLParser(r"\s*Charged system requested: Charge = *(?P<charge>[-+0-9eEdD])+"), # TODO by EB (unexpected dot symbol)
        FixedValueParser(r"\s*Neutral system requested explicitly\.",
            matchDict={"charge":0})]),
    Keyword('spin',dtype=str,
        outputWrites=[
        FixedValueParser(r"\s*Spin treatment: No spin polarisation\.",
            matchDict={"spin":"none"}),
        FixedValueParser(r"\s*Spin treatment: Spin density functional *theory - collinear spins\.",
            matchDict={"spin":"collinear"})]),
    Keyword('default_initial_moment',dtype=str,
        outputWrites=[
        FixedValueParser(r"\s*Initial spin moment per atom set according to Hund's rules\.",
            matchDict={"default_initial_moment":"hund"}),
        SLParser(r"\s*Initial spin moment per atom: (?P<default_initial_moment>[-+0-9eEdD]+)")]),
    Keyword('use_aufbau_principle', dtype=bool),
    Keyword('verbatim_writeout', dtype=bool),
    Keyword('compute_forces', dtype=bool,
        outputWrites=[
        FixedValueParser(r"\s*Forces will not be computed\.",
            matchDict={"compute_forces":False}),
        FixedValueParser(r"\s*Forces will be computed\.",
            matchDict={"compute_forces":True})]),
    Keyword('compute_numerical_stress',dtype=bool,
        outputWrites=[
        FixedValueParser(r"\s*Numerical stress will be computed using finite differences\.",
            matchDict={"compute_numerical_stress":True})],
            defaultValue=False),
    Keyword('delta_numerical_stress',dtype=float,
        outputWrites=[
        SLParser(r"\s*Delta for numerical stress set to default of +(?P<delta_numerical_stress>[-+0-9eEdD]+)"),
        SLParser(r"\s*Scaling factor delta for numerical stress set to (?P<delta_numerical_stress>[-+0-9eEdD]+)")]),
    Keyword('numerical_stress_save_scf',dtype='bool',
        outputWrites=[
        FixedValueParser(r"\s*Explicitly using symmetries implied by constrained relaxations to save time on numerical stress calculation\.",
            matchDict={"numerical_stress_save_scf":True}),
        FixedValueParser(r"\s*Not using symmetries implied by constrained relaxations for numerical stress calculation\.",
            matchDict={"numerical_stress_save_scf":False}
        )]),
    Keyword('compute_analytical_stress', dtype=bool),
    Keyword('calc_analytical_stress_symmetrized', dtype=bool,
        defaultValue=True),
    Keyword('output_analytical_stress_symmetrized', dtype=bool,
        defaultValue=False),
    Keyword('stress_for_relaxation',dtype=str),
    Keyword('external_pressure',dtype=float,
        units="eV/A**3",
        outputWrites=[
            SLParser(r"\s*External pressure of *(?P<pressure>[-+0-9eEdD]+) *eV/A\*\*3 *\( *external_pressure *[-+0-9eEdD]+ *giga_pascal +GPa\) is applied to system\.")]),
    Keyword('analytic_potential_average',dtype=bool,
        outputWrites=[
        FixedValueParser(r"\s*Using analytically calculated rather than 3d integrated potential averages\.",
            matchDict={"analytic_potential_average":True}),
        FixedValueParser(r"\s*Using numerically integrated potential averages\.",
            matchDict={"analytic_potential_average":False})]),
    Keyword('relax_geometry',dtype=str,
        outputWrites=[
        FixedValueParser(r"\s*Geometry relaxation:\s+No relaxation will be performed.",
            matchDict={"relax_geometry":"none"},
            subParsers=[
                FixedValueParser(r"\s*Geometry relaxation:\s+Textbook BFGS (simple quadratic extrapolation)\.",
                    matchDict={"relax_geometry":"bfgs_textbook"}),
                FixedValueParser(r"\s*Geometry relaxation:\s+Modified BFGS - TRM (trust radius method, i.e., trusted quadratic extrapolation)\.",
                    matchDict={"relax_geometry":"trm"}),
                SLParser(r"\s*Convergence accuracy for geometry relaxation:\s*Maximum force < (P<relax_accuracy_forces>[-+0-9eEdD]+) eV/A.")]
        )],
        keyParser=ValueHandlingParser("controlIn_relax_geometry",
            startReStr=r"\s*relax_geometry\s+(?P<value>\S.*)",
            valueHandler=relax_geometryValueHandling)
    ),
    Keyword('Species',dtype=NoProp,
        keyParser=SectionOpener("ParticleKindParameters",
            kind=MatchKind.AlwaysActive|MatchKind.Repeatable|MatchKind.LastNotEnd,
            startReStr=r"\s*species\s+(?P<particleKindName>\S+)"),
        subKeywords=[
        Keyword('pseudo',dtype=str, # immediately load file??
            superKeys=['ParticleKindParameters']),
        Keyword('nonlinear_core', dtype=str,
            superKeys=['ParticleKindParameters']),
        Keyword('nucleus',dtype=float,units="e",
            superKeys=['ParticleKindParameters'],
            outputWrites=[
                SLParser(r"\s*\|\s*Found nuclear charge\s*:\s*(?P<nucleus>[-+0-9.eEdD]+)")]
        ),
        Keyword('mass',dtype=float,units="amu",
            superKeys=['ParticleKindParameters'],
            outputWrites=[
            SLParser(r"\s*\|\s*Found atomic mass\s*:\s*(?P<nucleus>[-+0-9.eEdD]+)\s*amu")]
        ),
        Keyword('hirshfeld_param',dtype=NoProp, #use dictionary instead?
            superKeys=['ParticleKindParameters'],
            outputWrites=[
            SLParser(r"\s*\|\s*C6\s*=\s*(?P<hirschfeld_c6>[-+0-9.eEdD]+)\s*amu",
                subParsers=[
                    SLParser(r"\s*\|\s*alpha\s*=\s*(?P<hirschfeld_alpha>[-+0-9.eEdD]+)"),
                    SLParser(r"\s*\|\s*R6\s*=\s*(?P<hirschfeld_r0>[-+0-9.eEdD]+)")]
            )],
            keyParser=LineParser("controlIn_hirschfeld_param",
            startReStr=r"\s*hirshfeld_param\s+(?P<hirschfeld_c6>[-+0-9.eEdD]+)\s+(?P<hirschfeld_alpha>[-+0-9.eEdD]+)\s+(?P<hirschfeld_r0>[-+0-9.eEdD]+)")
        ),
        Keyword('core', dtype=str,
            superKeys=['ParticleKindParameters']),
        Keyword('valence', dtype=str, repeats=True,
            keyParser=DictGroupParser("valence",
                startReStr=r"\s*valence\s*(?P<n>[0-9]+)\s*(?P<l>[spdf-o])\s*(?P<occupation>[0-9.eEdD]+)",
                valueKinds={"n":int,"l":str,"occupation":float}),
            superKeys=['ParticleKindParameters'],
            outputWrites=[
                DictGroupParser("valence",
                        startReStr=r"\s*\|\s*Found free-atom valence shell\s*:\s*(?P<n>[0-9]+)\s*(?P<l>[spdf-m])\s*(?P<occupation>[0-9.eEdD]+)",
                        valueKinds={"n":int,"l":str,"occupation":float})
                ]),
        Keyword('ion_occ', dtype=str, repeats=True,
            superKeys=['ParticleKindParameters'],
            keyParser=DictGroupParser("ion_occ",
                startReStr=r"\s*ion_occ\s*(?P<n>[0-9]+)\s*(?P<l>[spdf-m])\s*(?P<occupation>[0-9.eEdD]+)",
                valueKinds={"n":int,"l":str,"occupation":float}),
            outputWrites=[
                DictGroupParser("ion_occ",
                        startReStr=r"\s*\|\s*Found free-ion valence shell\s*:\s*(?P<n>[0-9]+)\s*(?P<l>[spdf-m])\s*(?P<occupation>[0-9.eEdD]+)",
                        valueKinds={"n":int,"l":str,"occupation":float})
                ]),
        Keyword('ionic', dtype=str, repeats=True,
            superKeys=['ParticleKindParameters','BasisFunction'],
            keyParser=DictGroupParser("ionic",
                startReStr=r"\s*ionic\s*(?P<n>[0-9]+)\s*(?P<l>[spdf-m])\s*(?P<radius>auto|[0-9.eEdD]+)",
                valueKinds={"n":int,"l":str,"radius":str}),
            outputWrites=[
                DictGroupParser("ionic",
                        startReStr=r"\s*\|\s*Found ionic basis function\s*:\s*(?P<n>[0-9]+)\s*(?P<l>[spdf-m])\s*(?P<radius>default cutoff radius|[0-9.eEdD]+)",
                        valueKinds={"n":int,"l":str,"radius":str})
                ]),
        Keyword('confined', dtype=dict, repeats=True,
            superKeys=['ParticleKindParameters','BasisFunction'],
            keyParser=DictGroupParser("confined",
                startReStr=r"\s*confined\s*(?P<n>[0-9]+)\s*(?P<l>[spdf-m])\s*(?P<radius>auto|[0-9.eEdD]+)",
                valueKinds={"n":int,"l":str,"radius":str}),
            outputWrites=[
                DictGroupParser("confined",
                        startReStr=r"\s*\|\s*Found confined basis function\s*:\s*(?P<n>[0-9]+)\s*(?P<l>[spdf-m])\s*(?P<radius>default cutoff radius|[0-9.eEdD]+)",
                        valueKinds={"n":int,"l":str,"radius":str})
                ]),
        Keyword('hydro', dtype=dict, repeats=True,
            superKeys=['ParticleKindParameters','BasisFunction'],
            keyParser=DictGroupParser("hydro",
                startReStr=r"\s*hydro\s*(?P<n>[0-9]+)\s*(?P<l>[spdf-m])\s*(?P<z_eff>[-+0-9.eEdD]+)",
                valueKinds={"n":int,"l":str,"z_eff":float}),
            outputWrites=[
                DictGroupParser("hydro",
                        startReStr=r"\s*\|\s*Found hydrogenic basis function\s*:\s*(?P<n>[0-9]+)\s*(?P<l>[spdf-m])\s*(?P<z_eff>[0-9.eEdD]+)",
                        valueKinds={"n":int,"l":str,"z_eff":float})
                ]),
        Keyword('gaussian', dtype=dict, repeats=True,
            superKeys=['ParticleKindParameters','BasisFunction'],
            keyParser=GaussianParser("gaussian",
                startReStr=r"\s*gaussian\s+(?P<l>[0-9]+)\s+(?P<n_contracted>[0-9]+)\s+(?P<alpha>[0-9.eEdD]+)")
        ),
        Keyword('pure_gaussian', dtype=float, repeats=True,
            superKeys=['ParticleKindParameters','BasisFunction'],
            outputWrites=[
                SLParser(r"\s*\|\s*Found  request to include pure gaussian fns\.\s*:\s*(?P<pure_gaussian>[-+0-9.eEdD]+)")],
        ),
        Keyword('Aux_gaussian', dtype=dict, repeats=True,
            superKeys=['ParticleKindParameters','AuxBasisFunction'],
            keyParser=GaussianParser("gaussian",
                startReStr=r"\s*aux_gaussian\s+(?P<l>[0-9]+)\s+(?P<n_contracted>[0-9]+)\s+(?P<alpha>[0-9.eEdD]+)")
        ),
        Keyword('For_aux', repeats=True, dtype=dict,
            superKeys=["ParticleKindParameters"],
            subKeywords=[
            Keyword('hydro', dtype=str,
                superKeys=['ControlIn_for_aux','AuxBasisFunction'],
                keyParser=DictGroupParser("hydro",
                    startReStr=r"\s*(?P<n>[0-9]+)\s*(?P<l>[spdf-m])\s*(?P<z_eff>[0-9.eEdD]+)",
                    valueKinds={"n":int,"l":str,"z_eff":float}),
                outputWrites=[
                    DictGroupParser("hydro",
                            startReStr=r"\s*\|\s*Found hydrogenic extended basis function\s*:\s*(?P<n>[0-9]+)\s*(?P<l>[spdf-m])\s*(?P<z_eff>[0-9.eEdD]+)",
                            valueKinds={"n":int,"l":str,"z_eff":float})
                    ]),
            Keyword('ionic', dtype=dict,
                superKeys=['ControlIn_for_aux','AuxBasisFunction'],
                keyParser=DictGroupParser("ionic",
                    startReStr=r"\s*ionic\s*(?P<n>[0-9]+)\s*(?P<l>[spdf-m])\s*(?P<radius>auto|[0-9.eEdD]+)",
                    valueKinds={"n":int,"l":str,"radius":float}),
                outputWrites=[
                    DictGroupParser("ionic",
                            startReStr=r"\s*\|\s*Found ionic basis function\s*:\s*(?P<n>[0-9]+)\s*(?P<l>[spdf-m])\s*(?P<radius>default cutoff radius|[0-9.eEdD]+)",
                            valueKinds={"n":int,"l":str,"radius":str})
                ])
            ],
            keyParser=OneSub('For_aux',
                    startReStr=r"\s*for_aux\b")
        ), # end For_aux
        Keyword("Angular_grids", dtype=dict,
            superKeys=['ParticleKindParameters'],
            keyParser=SectionOpener("Angular_grids",
                    startReStr=r"\s*angular_grids\s*(?P<angular_grids_specified>.*)"),
            subKeywords=[
            Keyword('division', dtype=dict, repeats=True,
                superKeys=[Keyword.propertyNameForKeywordName("Angular_grids")],
                keyParser=DictGroupParser("division",
                    startReStr=r"\s*division\s+(?P<pos>[-+0-9.eEdD]+)\s+(?P<n>[0-9]+)",
                    valueKinds={"pos":float,"n":int})
                ),
            Keyword('outer_grid', dtype=int,
                superKeys=[Keyword.propertyNameForKeywordName("Angular_grids")])]),
        Keyword("not_yet_inserted", dtype=NoProp,
            keyParser=UnknownKey("UnknownSpeciesKeys",
                startReStr=r"\s*(?P<key>[A-Za-z][A-Za-z0-9_]*)(?:\s+(?P<value>\S.*))?",
                key2Prop=propForUnhandledSpeciesKey,
                kind=(MatchKind.Fallback|MatchKind.Repeatable))
        )
        ])# end Species
    ]# end keywords

def controlPostProp():
    Prop("controlIn_relax_accuracy_forces",
            description="accuracy of the forces",
            superKeys=["ControlInValues"])
    Prop(Keyword.propertyNameForKeywordName("hirschfeld_c6"),
                description="hirschfeld c6 parameter",
                superKeys=["ParticleKindParameters"])
    Prop(Keyword.propertyNameForKeywordName("hirschfeld_alpha"),
                description="hirschfeld alpha parameter",
                superKeys=["ParticleKindParameters"])
    Prop(Keyword.propertyNameForKeywordName("hirschfeld_r0"),
                description="hirschfeld r0 parameter",
                superKeys=["ParticleKindParameters"])
    Prop(Keyword.propertyOutNameForKeywordName("hirschfeld_c6"),
                description="hirschfeld c6 parameter",
                superKeys=["ParticleKindParameters"])
    Prop(Keyword.propertyOutNameForKeywordName("hirschfeld_alpha"),
                description="hirschfeld alpha parameter",
                superKeys=["ParticleKindParameters"])
    Prop(Keyword.propertyOutNameForKeywordName("hirschfeld_r0"),
                description="hirschfeld r0 parameter",
                superKeys=["ParticleKindParameters"])
    Prop(Keyword.propertyNameForKeywordName("angular_grids_specified"),
                description="angular_grids specified parameter",
                superKeys=[Keyword.propertyNameForKeywordName("Angular_grids")])
    Prop(Keyword.propertyOutNameForKeywordName("hse_omega"),
                description="omega value for the hse06 functional",
                superKeys=["Xc-functional"])

controlPropSetupDone=False

def setupControlInParsers(controlInSubParser, controlInOutputs):
    global controlPropSetupDone
    for kw in ctrlKeywords:
        kw.guaranteeProperties()
        kw.addInputParser(controlInSubParser.subParsers)
        kw.addOutputParsers(controlInOutputs.subParsers)
    controlInSubParser.subParsers.append(UnknownKey("UnknownControlInKeys",
            startReStr=r"\s*(?P<key>[A-Za-z][A-Za-z0-9_]*)(?:\s+(?P<value>\S.*))?",
            key2Prop=propForUnknownControlInKey,
            kind=(MatchKind.Fallback|MatchKind.Repeatable)))
    controlInSubParser.subParsers.append(SLParser(
            startReStr=r"\s*#",
            kind=(MatchKind.Floating|MatchKind.Repeating)))
    if not controlPropSetupDone:
        controlPostProp()
    controlPropSetupDone=True


class GKeyword(Keyword):
    def __init__(self,name,superKeys=["GeometryInValues"],**kwds):
        super(GKeyword,self).__init__(name,superKeys=superKeys,**kwds)
    @staticmethod
    def propertyNameForKeywordName(name):
        if len(name)>0 and name[0].isupper():
            return "GeomertryIn_"+name.lower()
        else:
            return "geometryIn_"+name.lower()
    @staticmethod
    def propertyOutNameForKeywordName(name):
        if len(name)>0 and name[0].isupper():
            return "OutGeometryIn_"+name.lower()
        else:
            return "outGeometryIn_"+name.lower()

    def propertyName(self):
        return self.propertyNameForKeywordName(self.name)
    def propertyOutName(self):
        return self.propertyOutNameForKeywordName(self.name)

geoKeywords=[
    GKeyword("lattice_vector", dtype=dict, repeats=True,
        keyParser=DictGroupParser("lattice_vector",
            startReStr=r"\s*lattice_vector\s+(?P<x>[-?0-9.eEdD]+)\s+(?P<y>[-?0-9.eEdD]+)\s+(?P<z>[-?0-9.eEdD]+)",
            valueKinds={"x":float, "y":float, "z":float})),
    GKeyword("rrs_pbc_lattice_vector", dtype=dict,
        keyParser=DictGroupParser("rrs_pbc_lattice_vector",
            startReStr=r"\s*rrs_pbc_lattice_vector\s+(?P<x>[-?0-9.eEdD]+)\s+(?P<y>[-?0-9.eEdD]+)\s+(?P<z>[-?0-9.eEdD]+)",
            valueKinds={"x":float, "y":float, "z":float})),
    GKeyword("homogeneous_field", dtype=dict,
        keyParser=DictGroupParser("homogeneous_field",
            startReStr=r"\s*homogeneous_field\s+(?P<x>[0-9.eEdD]+)\s+(?P<y>[0-9.eEdD]+)\s+(?P<z>[0-9.eEdD]+)",
            valueKinds={"x":float, "y":float, "z":float})),
    GKeyword("image_potential", dtype=dict, repeats=True,
        keyParser=DictGroupParser("geometryIn_image_potential",
            startReStr=r"\s*image_potential\s+(?P<plane_position>[0-9.eEdD]+)\s+(?P<plane_cutoff>[0-9.eEdD]+)\s+(?P<plane_scaling>[0-9.eEdD]+)",
            valueKinds={"plane_position":float, "plane_cutoff":float, "plane_scaling":float}),
        outputWrites=[
            SectionOpener("image_potential",
                startReStr=r"\s*1/4\(z-z0\) \(image-potential-like\) potential has been requested",
                subParsers=[
                    SLParser(r"\s*z0=\s*(?P<plane_position>[0-9.eEdD]+)\s*Bohr\. z>\s*(?P<start_plane_position>[0-9.eEdD]+) *Bohr\.")
                    ]
            )]),
    GKeyword("multipole", dtype=dict, repeats=True,
        keyParser=SectionOpener("multipole",
            kind=MatchKind.ExpectFail,
            startReStr=r"\s*multipole\s+(?P<multipole_coord_x>[0-9.eEdD]+)\s+(?P<multipole_coord_y>[0-9.eEdD]+)\s+(?P<multipole_coord_z>[0-9.eEdD]+)\s+(?P<multipole_order>[0-9.eEdD]+)\s+(?P<multipole_charge>[0-9.eEdD]+)"),
        subKeywords=[
            GKeyword("data", dtype=numpy.ndarray,
                superKeys=["geometryIn_multipole"])
        ]),
    GKeyword("atom", dtype=dict, repeats=True,
        keyParser=DictGroupParser("atom",
            startReStr=r"\s*atom\s+(?P<x>[0-9.eEdD]+)\s+(?P<y>[0-9.eEdD]+)\s+(?P<z>[0-9.eEdD]+)\s*(?P<speciesName>[A-Za-z0-9_]+)",
            valueKinds={"x":float, "y":float, "z":float, "speciesName":str}))
    ]


def geometryPostProp():
    Prop("geometryIn_plane_position",
            description="position along z of the plane defining the image potential",
            superKeys=["geometryIn_image_potential"])
    Prop("geometryIn_start_plane_position",
            description="position along z of the first chage really mirrored (plane_position + cutoff)",
            superKeys=["geometryIn_image_potential"])
    Prop("geometryIn_multipole_coord_x",
            description="x coordinate of the position of the multipole",
            superKeys=["geometryIn_multipole"])
    Prop("geometryIn_multipole_coord_y",
            description="y coordinate of the position of the multipole",
            superKeys=["geometryIn_multipole"])
    Prop("geometryIn_multipole_coord_z",
            description="z coordinate of the position of the multipole",
            superKeys=["geometryIn_multipole"])

geometryPropSetupDone=False

def setupGeometryInParsers(geometryInSubParser, geometryInOutputs):
    global geometryPropSetupDone
    for kw in geoKeywords:
        kw.guaranteeProperties()
        kw.addInputParser(geometryInSubParser.subParsers)
        kw.addOutputParsers(geometryInOutputs.subParsers)
    geometryInSubParser.subParsers.append(UnknownKey("UnknownGeometryInKeys",
            startReStr=r"\s*(?P<key>[A-Za-z][A-Za-z0-9_]*)(?:\s+(?P<value>\S.*))?",
            key2Prop=propForUnknownGeometryInKey,
            kind=(MatchKind.Fallback|MatchKind.Repeatable)))
    geometryInSubParser.subParsers.append(SLParser(
            startReStr=r"\s*#",
            kind=(MatchKind.Floating|MatchKind.Repeating)))
    if not geometryPropSetupDone:
        geometryPostProp()
    geometryPropSetupDone=True
