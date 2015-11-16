
# Author: Fawzi Mohamed

import numpy
import os, sys
import re
import traceback
import datetime
import logging
import h5py

import setup_paths
import FhiAimsControlInParser
from LineParsers import *
import TrajUtils

# 2010 CODATA recommended values
eV = 1.0/27.21138505 # http://physics.nist.gov/cgi-bin/cuu/Convert?exp=0&num=1&From=hr&To=ev&Action=Only+show+factor
angstrom = 1.0/0.52917721092 # http://physics.nist.gov/cgi-bin/cuu/Value?bohrrada0

nLog = logging.getLogger("nomad")


properties = [
    Prop(name="ProgramInfo",
        description="""Information on the program that generated the data"""),
    Prop(name="programName",
        description="""Name of the program that generated the data""",
        superKeys=["ProgramInfo"]),
    Prop(name='programVersion',
        description="""Version of the program that was used""",
        superKeys=["ProgramInfo"]),
    Prop(name="programCompilationDate",
        description="""Date the program was compiled""",
        superKeys=["ProgramInfo"]),
    Prop(name="programCompilationTime",
        description="""Time the program was compiled""",
        superKeys=["ProgramInfo"]),
    Prop(name="programCompilationHost",
        description="""Host on which the program was compiled""",
        superKeys=["ProgramInfo"]),

    Prop(name="AccessoryRunInfo",
        description="""Information on the run which *in theory* should have no influence on the results""",
        kind="section"),

    Prop(name="TimeInfo",
        description="""Information on date, and timings""",
        superKeys=["AccessoryRunInfo"]),
    Prop(name="programExecutionDate",
        description="""Date of execution of the program""",
        repeats=True, superKeys=["TimeInfo"]),
    Prop(name="programExecutionTime",
        description="""Time of execution of the program""",
        repeats=True, superKeys=["TimeInfo"]),
    Prop(name="t0Cpu1",
        description="""Time zero on CPU 1""",
        superKeys=["TimeInfo"], units="s"),
    Prop(name="t0Wall",
        description="""Internal wall clock time zero""",
        superKeys=["TimeInfo"], units="s"),

    Prop(name="ParallelizationInfo",
        description="""Information on the parallelzation of the program""",
        superKeys=["AccessoryRunInfo"]),
    Prop(name="parallelTasksAssignement",
        description="""Hosts that did run this simulation""",
        superKeys=["ParallelizationInfo"]),
    Prop(name="numberOfTasks",
        description="""Number of parallel tasks used""",
        superKeys=["ParallelizationInfo"]),

    Prop(name="SimulationParameters",
        description="parameters that control the simulation"),
    Prop(name="Xc-functional",
        description="exchange correlation functional realted information",
        superKeys=["SimulationParameters"]),
    Prop(name="xc-functional-code",
        description="exchange correlation functional in short form",
        superKeys=["Xc-functional"]),
    Prop(name="ControlInValues",
        description="values of the control.in file",
        superKeys=["SimulationParameters"]),
    Prop(name="UnknownControlInKeys",
        description="control.in keys that were unknown to the normalization script",
        superKeys=["ControlInValues"]),
    Prop(name="postHfMethod",
        description="post hartree fock method used",
        superKeys=["SimulationParameters"]),

    Prop(name="GeometryInValues",
        description="values of the geometry.in file",
        superKeys=["SimulationParameters"]),
    Prop(name="UnknownGeometryInKeys",
        description="geometry.in keys that were unknown to the normalization script",
        superKeys=["GeometryInValues"]),

    Prop(name="ConfigurationProperties",
        description="""Properties connected with a configuration"""),
    Prop(name="CoreConfiguration",
        description="""Properties actually defining the current configuration""",
        superKeys=["ConfigurationProperties"], contextName="CoreConfiguration"),
    Prop(name="particleKindNames", repeats=True, dtype=str,
        description="""Names of the kinds (species) in a configuration.
        A value is valid from the definition point for all subsequent values until redefined
        (i.e. if not changing it can be defined just once at the beginning).""",
        superKeys=["CoreConfiguration"]),
    Prop(name="particlePositions", repeats=True, dtype=float, units="Bohr",
        description="""Positions of the particles (this defines a configuration and is required).""",
        superKeys=["CoreConfiguration"]),
    Prop(name="cell", repeats=True, dtype=float, units="Bohr",
        description="""Periodic cell given as the three basis vectors.
        A value is valid from the definition point for all subsequent values until redefined
        (i.e. if not changing it can be defined just once at the beginning).""",
        superKeys=["CoreConfiguration"]),
    Prop(name="periodicDimensions", repeats=True, dtype=bool,
        description="""Which of the basis vectors use periodic boundary conditions.
        A value is valid from the definition point for all subsequent values until redefined
        (i.e. if not changing it can be defined just once at the beginning).""",
        superKeys=["CoreConfiguration"]),
    Prop("particlePos", repeats=True, dtype=dict,
        superKeys=["CoreConfiguration"],
        description="temp property to keep particle position", contextName="CoreConfiguration"),

    Prop(name="Energy",
        description="Some energy value"),
    Prop(name="PartialEnergyComponent",
        description="An energy component of the non final, non converged energy",
        superKeys=["Energy","ConfigurationProperties"]),
    Prop(name="EnergyComponent",
        description="An energy component of the final, converged energy",
        superKeys=["PartialEnergyComponent"]),
    Prop(name="PartialTotalEnergy",
        description="A total energy that is not yet converged",
        superKeys=["PartialEnergyComponent"]),
    Prop(name="TotalEnergy",
        description="Some form of final, converged total energy",
        superKeys=["PartialTotalEnergy"]),
    Prop(name="PartialDftEnergy",
        description="A partial (non final, non converged) energy at DFT (or HartreeFock) level",
        superKeys=["PartialTotalEnergy"]),
    Prop(name="DftEnergy",
        description="A total (final, converged) energy at DFT (or HartreeFock) level",
        superKeys=["PartialDftEnergy","TotalEnergy"]),

    Prop(name="PartialErrorEstimate",
        description="Some estimate of the error of the non coverged value",
        superKeys=[]),
    Prop(name="ErrorEstimate",
        description="Some estimate of the error on the converged (final) value",
        superKeys=["PartialErrorEstimate"]),

    Prop(name="ConservedQuantity",
        description="A quantity that is preserved by the evolution (for example kinetic+potential energy during NVE"),

    Prop(name="ScfInfo", repeats=True,
        description="""Information on the scf procedure""",
        superKeys=["ConfigurationProperties"],
        contextName="ScfInfo"),
    Prop(name="startDateScf0", dtype=str,
        description="""Starting date of the scf procedure""",
        superKeys=["ScfInfo"],
        contextName="ScfInfo"),
    Prop(name="startTimeScf0", dtype=str,
        description="""Starting time of the scf procedure""",
        superKeys=["ScfInfo"],
        contextName="ScfInfo"),
    Prop(name="evSumScf0", dtype=float,
        description="Sum of the eigenvalues at the start of the scf",
        superKeys=["ScfInfo","PartialEnergyComponent"],
        contextName="ScfInfo", units="Hartree"),
    Prop(name="xcEnergyScf0", dtype=float,
        description="",
        superKeys=["ScfInfo","PartialEnergyComponent"],
        contextName="ScfInfo", units="Hartree"),
    Prop(name="xcPotentialEnergyScf0", dtype=float,
        description="",
        superKeys=["ScfInfo","PartialEnergyComponent"],
        contextName="ScfInfo", units="Hartree"),
    Prop(name="eStaticEnergyScf0", dtype=float,
        description="",
        superKeys=["ScfInfo","PartialEnergyComponent"],
        contextName="ScfInfo", units="Hartree"),
    Prop(name="hartreeEnergyScf0", dtype=float,
        description="",
        superKeys=["ScfInfo","PartialEnergyComponent"],
        contextName="ScfInfo", units="Hartree"),
    Prop(name="entropyCorrectionScf0", dtype=float,
        description="",
        superKeys=["ScfInfo","PartialEnergyComponent"],
        contextName="ScfInfo", units="Hartree"),
    Prop(name="totalDftEnergyScf0", dtype=float,
        description="",
        superKeys=["ScfInfo", "PartialDftEnergy"],
        contextName="ScfInfo", units="Hartree"),
    Prop(name="totalDftEnergyT0Scf0", dtype=float,
        description="",
        superKeys=["ScfInfo", "PartialDftEnergy"],
        contextName="ScfInfo", units="Hartree"),
    Prop(name="eletronicFreeEnergyScf0", dtype=float,
        description="",
        superKeys=["ScfInfo","PartialEnergyComponent"],
        contextName="ScfInfo", units="Hartree"),
    Prop(name="electronicKineticEnergyScf0", dtype=float,
        description="",
        superKeys=["ScfInfo","PartialEnergyComponent"],
        contextName="ScfInfo", units="Hartree"),
    Prop(name="electrostaticEnergyScf0", dtype=float,
        description="",
        superKeys=["ScfInfo","PartialEnergyComponent"],
        contextName="ScfInfo", units="Hartree"),
    Prop(name="hartreeEnergyErrorScf0", dtype=float,
        description="",
        superKeys=["ScfInfo","PartialEnergyComponent"],
        contextName="ScfInfo", units="Hartree"),
    Prop(name="sumEigenvaluesPerAtomScf0", dtype=float,
        description="",
        superKeys=["ScfInfo","PartialEnergyComponent"],
        contextName="ScfInfo", units="Hartree"),
    Prop(name="totalEnergyT0PerAtomScf0", dtype=float,
        description="",
        superKeys=["ScfInfo","PartialEnergyComponent"],
        contextName="ScfInfo", units="Hartree"),
    Prop(name="electronicFreeEnergyPerAtomScf0", dtype=float,
        description="",
        superKeys=["ScfInfo","PartialEnergyComponent"],
        contextName="ScfInfo", units="Hartree"),

    Prop(name="ScfIteration", repeats=True,
        description="represents an scf iteration",
        superKeys=["ScfInfo"],
        contextName="ScfInfo"),
    Prop(name="scfIterationNr", dtype=int,
        description="The number of this iteration",
        superKeys=["ScfIteration"],
        contextName="ScfInfo"),
    Prop(name="startDateScfIter", dtype=str,
        description="""Starting date of the scf procedure""",
        superKeys=["ScfIteration"],
        contextName="ScfInfo"),
    Prop(name="startTimeScfIter", dtype=str,
        description="""Starting time of the scf procedure""",
        superKeys=["ScfIteration"],
        contextName="ScfInfo"),
    Prop(name="evSumScfIter", dtype=float,
        description="Sum of the eigenvalues at the start of the scf",
        superKeys=["ScfIteration","PartialEnergyComponent"],
        contextName="ScfInfo", units="Hartree"),
    Prop(name="xcEnergyScfIter", dtype=float,
        description="",
        superKeys=["ScfIteration","PartialEnergyComponent"],
        contextName="ScfInfo", units="Hartree"),
    Prop(name="xcPotentialEnergyScfIter", dtype=float,
        description="",
        superKeys=["ScfIteration","PartialEnergyComponent"],
        contextName="ScfInfo", units="Hartree"),
    Prop(name="eStaticEnergyScfIter", dtype=float,
        description="",
        superKeys=["ScfIteration","PartialEnergyComponent"],
        contextName="ScfInfo", units="Hartree"),
    Prop(name="hartreeEnergyScfIter", dtype=float,
        description="",
        superKeys=["ScfIteration","PartialEnergyComponent"],
        contextName="ScfInfo", units="Hartree"),
    Prop(name="entropyCorrectionScfIter", dtype=float,
        description="",
        superKeys=["ScfIteration","PartialEnergyComponent"],
        contextName="ScfInfo", units="Hartree"),
    Prop(name="totalDftEnergyScfIter", dtype=float,
        description="",
        superKeys=["ScfIteration", "PartialDftEnergy"],
        contextName="ScfInfo", units="Hartree"),
    Prop(name="totalDftEnergyT0ScfIter", dtype=float,
        description="",
        superKeys=["ScfIteration", "PartialDftEnergy"],
        contextName="ScfInfo", units="Hartree"),
    Prop(name="eletronicFreeEnergyScfIter", dtype=float,
        description="",
        superKeys=["ScfIteration","PartialEnergyComponent"],
        contextName="ScfInfo", units="Hartree"),
    Prop(name="electronicKineticEnergyScfIter", dtype=float,
        description="",
        superKeys=["ScfIteration","PartialEnergyComponent"],
        contextName="ScfInfo", units="Hartree"),
    Prop(name="electrostaticEnergyScfIter", dtype=float,
        description="",
        superKeys=["ScfIteration","PartialEnergyComponent"],
        contextName="ScfInfo", units="Hartree"),
    Prop(name="hartreeEnergyErrorScfIter", dtype=float,
        description="",
        superKeys=["ScfIteration","PartialEnergyComponent"],
        contextName="ScfInfo", units="Hartree"),
    Prop(name="sumEigenvaluesPerAtomScfIter", dtype=float,
        description="",
        superKeys=["ScfIteration","PartialEnergyComponent"],
        contextName="ScfInfo", units="Hartree"),
    Prop(name="totalEnergyT0PerAtomScfIter", dtype=float,
        description="",
        superKeys=["ScfIteration","PartialEnergyComponent"],
        contextName="ScfInfo", units="Hartree"),
    Prop(name="electronicFreeEnergyPerAtomScfIter", dtype=float,
        description="",
        superKeys=["ScfIteration","PartialEnergyComponent"],
        contextName="ScfInfo", units="Hartree"),

    Prop(name="chargeDensityChangeScfIter", dtype=float,
        description="Change ot the charge density during an scf iteration",
        superKeys=["ScfIteration","PartialErrorEstimate"],
        contextName="ScfInfo", units="e"),
    Prop(name="sumEigChangerScfIter", dtype=float,
        description="Change of the sum of the eigenvalues during an scf iteration",
        superKeys=["ScfIteration","PartialErrorEstimate"],
        contextName="ScfInfo", units="Hartree"),
    Prop(name="totalEnergyChangeScfIter", dtype=float,
        description="Change of total energy during an scf iteration",
        superKeys=["ScfIteration","PartialErrorEstimate"],
        contextName="ScfInfo", units="Hartree"),

    Prop(name="totalDftEnergy", dtype=float, kind=PropertyKind.configProperty,
        description="Converged total dft energy",
        superKeys=["ConfigurationProperties","DftEnergy"],
        units="Hartree"),
    Prop(name="totalDftEnergyT0", dtype=float, kind=PropertyKind.configProperty,
        description="Converged dft energy with T0 correction",
        superKeys=["ConfigurationProperties","DftEnergy"],
        units="Hartree"),
    Prop(name="dftElectronicFreeEnergy", dtype=float, kind=PropertyKind.configProperty,
        description="Dft electronic free energy",
        superKeys=["ConfigurationProperties","DftEnergy"],
        units="Hartree"),

    Prop(name="nScfDft", dtype=int,
        description="number of scf iterations at Dft level",
        superKeys=["ScfInfo"]),

    Prop(name="exactExchangeEnergy", dtype=float, kind=PropertyKind.configProperty,
        description="Converged exact exchange energy",
        superKeys=["ConfigurationProperties", "EnergyComponent"],
        units="Hartree"),
    Prop(name="xEnergyDft", dtype=float, kind=PropertyKind.configProperty,
        description="Converged exchange part of the dft xc functional",
        superKeys=["ConfigurationProperties", "EnergyComponent"],
        units="Hartree"),
    Prop(name="cEnergyDft", dtype=float, kind=PropertyKind.configProperty,
        description="Converged correlation part of the dft xc functional",
        superKeys=["ConfigurationProperties", "EnergyComponent"],
        units="Hartree"),
    Prop(name="xEnergyLda", dtype=float, kind=PropertyKind.configProperty,
        description="exchange part of the lda xc functional using the self consistent density of the target functional[correct???]",
        superKeys=["ConfigurationProperties", "EnergyComponent"],
        units="Hartree"),
    Prop(name="cEnergyLda", dtype=float, kind=PropertyKind.configProperty,
        description="correlation part of the lda xc functional using the self consistent density of the target functional[correct???]",
        superKeys=["ConfigurationProperties", "EnergyComponent"],
        units="Hartree"),
    Prop(name="hfEnergy", dtype=float, kind=PropertyKind.configProperty,
        description="Converged Hartree-Fock energy",
        superKeys=["ConfigurationProperties", "DftEnergy"],
        units="Hartree"),
    Prop("mp2EnergyCorrection", dtype=float, kind=PropertyKind.configProperty,
        description="Converged Mp2 energy correction",
        superKeys=["ConfigurationProperties","EnergyComponent"],
        units="Hartree"),
    Prop("mp2Energy", dtype=float, kind=PropertyKind.configProperty,
        description="Converged total MP2 energy",
        superKeys=["ConfigurationProperties","TotalEnergy"],
        units="Hartree"),
    Prop("finalTotalDftEnergy", dtype=float, kind=PropertyKind.configProperty,
        description="Last converged total dft energy",
        superKeys=["TotalEnergy"],
        units="Hartree"),
    Prop("finalTotalDftEnergyT0", dtype=float, kind=PropertyKind.configProperty,
        description="Last converged total dft energy corrected for T->0",
        superKeys=["TotalEnergy"],
        units="Hartree"),
    Prop("finalCorrectedEnergy", dtype=float, kind=PropertyKind.configProperty,
        description="Last converged total energy with post scf corrections (which ones depend on controlIn_total_energy_method)",
        superKeys=["TotalEnergy"],
        units="Hartree"),
    Prop("vanDerWaalsEnergy", dtype=float, kind=PropertyKind.configProperty,
        description="Converged van der Waals energy",
        superKeys=["ConfigurationProperties", "PartialEnergyComponent"]),
    Prop("vanDerWaalsEnergyTS", dtype=float, kind=PropertyKind.configProperty,
        description="Converged van der Waals energy using the Tkatchenko Scheffler method",
        superKeys=["vanDerWaalsEnergy"]),

    Prop("mp2Time", dtype=float, repeats=True,
        description="time needed for an mp2 calculation",
        superKeys=["TimeInfo"], units="s"),

    Prop(name="SpeciesParameters", repeats=True,
        description="parameters of one atom kind",
        superKeys=["SimulationParameters"]),

    Prop(name="ParticleKindParameters", repeats=True,
        description="parameters of one particle kind",
        superKeys=["SimulationParameters"]),
    Prop(name="particleKindName",
        description="Name (identifier) of one particle kind",
        superKeys=["ParticleKindParameters"]),
    Prop(name="ParticleInfo", repeats=True,
        description="a particle information",
        superKeys=["ParticleKindParameters"]),
    Prop(name="BasisFunction", repeats=True,
        description="A basis function",
        superKeys=["ParticleKindParameters"]
        ),
    Prop(name="AuxBasisFunction", repeats=True,
        description="An auxiliary basis function",
        superKeys=["ParticleKindParameters"]),

    Prop(name="eVHaValue",
        description="Hartree in eV",
        superKeys=["AccessoryRunInfo"]),# see this as parameter?

    Prop(name="band_k_points", repeats=True,
        description="k-Points of the band",
        superKeys=["ConfigurationProperties"]),
    Prop(name="band_occupations", repeats=True,
        description="occupations of the band",
        superKeys=["ConfigurationProperties"]),
    Prop(name="band_energies", repeats=True,
        description="energies of the band",
        superKeys=["ConfigurationProperties"]),
    Prop(name="band_segm_labels", repeats=True, dtype=str,
        description="labels for each segment of the band",
        superKeys=["ConfigurationProperties"]),
    Prop(name="band_segm_start_end", repeats=True,
        description="start and end points of the band segments",
        superKeys=["ConfigurationProperties"]),

    Prop(name="has_GW", kind=PropertyKind.configProperty,
        description="whether the calculation includes GW treatment",
        superKeys=["AccessoryRunInfo"])
]

controlInSubParser=GroupSection("controlInSubParser",kind=MatchKind.Sequenced|MatchKind.LastNotEnd)
controlInOutputs=GroupSection("ControlInOutputs",kind=MatchKind.Sequenced|MatchKind.LastNotEnd|MatchKind.ExpectFail)
geometryInSubParser=GroupSection("geometryInSubParser",kind=MatchKind.Sequenced|MatchKind.LastNotEnd)
geometryInOutputs=GroupSection("GeometryInOutputs",kind=MatchKind.Sequenced|MatchKind.LastNotEnd|MatchKind.ExpectFail)

sections = [
    RootSection(name='newRun',
        startReStr=r"(?m)^ *Invoking FHI-aims \.\.\. *$",
        kind=MatchKind.Sequenced | MatchKind.Repeating | MatchKind.ExpectFail,

        subParsers = [
        GroupSection("ProgramHeader",
            subParsers=[
            SLParser(r" *Version +(?P<programVersion>[0-9a-zA-Z_.]*)"),
            SLParser(r" *Compiled on +(?P<programCompilationDate>[0-9/]+) at (?P<programCompilationTime>[0-9:]+) *on host +(?P<programCompilationHost>[-a-zA-Z0-9._]+)", kind=MatchKind.ExpectFail),
            SkipSome(),
            SLParser(r" *Date *: *(?P<programExecutionDate>[-.0-9/]+) *, *Time *: *(?P<programExecutionTime>[-+0-9.EDed]+)"),
            SLParser(r" *Time zero on CPU 1 *: *(?P<t0Cpu1>[-+0-9.EeDd]+) *s\."),
            SLParser(r" *Internal wall clock time zero *: *(?P<t0Wall>[-+0-9.EeDd]+) *s\."),
            LineParser("nParallelTasks", kind=MatchKind.ExpectFail,
                startReStr=r" *Using *(?P<numberOfTasks>[0-9]+) *parallel tasks\.",
                subParsers=[
                    CollectDict(name="parallelTasksAssignement",
                        startReStr=r" *Task *(?P<key>[0-9]+) *on host *(?P<value>[-a-zA-Z0-9._]+) *reporting\.")
                    ]
            )]
        ),
        SubFileParser('controlInParser',
            startReStr=r"\s*Parsing control\.in *(?:(?P<notInline>\.\.\.)|\(first pass over file, find array dimensions only\)\.)",
            preHeader=[
                r"(?i) *The contents of control\.in will be repeated verbatim below",
                r"(?i) *unless switched off by setting 'verbatim_writeout \.false\.' \.",
                r"(?i) *in the first line of control\.in *\.",
                r" *-{10}-*"],
            postHeader=[r" *-{10}-*"],
            extFilePath="control.in",
            firstSubFileParser=controlInSubParser
        ),
        controlInOutputs,
        SubFileParser('GeometryInParser',
            startReStr=r"\s*Parsing geometry\.in *(?:(?P<notInline>\.\.\.)|\(first pass over file, find array dimensions only\)\.)",
            preHeader=[
                r"(?i) *The contents of geometry\.in will be repeated verbatim below",
                r"(?i) *unless switched off by setting 'verbatim_writeout \.false\.' \.",
                r"(?i) *in the first line of geometry\.in *\.",
                r" *-{10}-*"],
            postHeader=[r" *-{10}-*"],
            extFilePath="geometry.in",
            firstSubFileParser=geometryInSubParser
        ),
        geometryInOutputs,
        ConfigParser('CoreConfiguration', kind=MatchKind.Sequenced|MatchKind.Repeatable,
            startReStr=r"\s*Input geometry\s*:",
            subParsers=[
            SLParser(name='NoCellParser', kind=MatchKind.Sequenced|MatchKind.ExpectFail,
                startReStr=r"\s*\|\s*No unit cell requested\."),
                # matchDict={"configCell":None}), # cell to do convert previous to FixedValueParser
            SLParser(r"\s*\|\s*Atomic structure\s*:"),
            SLParser(r"\s*\|\s*Atom\s+x\s*\[A\]\s*y\s*\[A\]\s+z\s*\[A\]"),
            DictGroupParser("particlePos", kind=MatchKind.Sequenced|MatchKind.Repeating,
                startReStr=r"\s*\|\s*(?P<iParticle>[0-9]+)\s*:\s*Species\s*(?P<kindName>\S+)\s+(?P<x>[-+0-9.eEdD]+)\s+(?P<y>[-+0-9.eEdD]+)\s+(?P<z>[-+0-9.eEdD]+)",
                valueKinds={'iParticle':int, 'name':str, 'x':float, 'y':float, 'z':float}),
            ]),
        ScfParser("ScfInfo", kind=MatchKind.Sequenced|MatchKind.Repeating|MatchKind.ExpectFail,
            startReStr=r"\s*Begin self-consistency loop: Initialization\.",
            subParsers=[
            SLParser(r" *Date *: *(?P<startDateScf0>[-.0-9/]+) *, *Time *: *(?P<startTimeScf0>[-+0-9.EDed]+)"),
            LineParser("TotalEnergyScf0",
                startReStr=r"\s*Total energy components:",
                subParsers=[
                SLParser(r"\s*\|\s*Sum of eigenvalues\s*:\s*(?P<evSumScf0>[-+0-9.eEdD]+) *Ha\s*[-+0-9.eEdD]+ *eV"),
                SLParser(r"\s*\|\s*XC energy correction\s*:\s*(?P<xcEnergyScf0>[-+0-9.eEdD]+) *Ha\s*[-+0-9.eEdD]+ eV"),
                SLParser(r"\s*\|\s*XC potential correction\s*:\s*(?P<xcPotentialEnergyScf0>[-+0-9.eEdD]+) *Ha\s*[-+0-9.eEdD]+ *eV"),
                SLParser(r"\s*\|\s*Free-atom electrostatic energy\s*:\s*(?P<eStaticEnergyScf0>[-+0-9.eEdD]+) *Ha\s*[-+0-9.eEdD]+ *eV"),
                SLParser(r"\s*\|\s*Hartree energy correction\s*:\s*(?P<hartreeEnergyScf0>[-+0-9.eEdD]+) *Ha\s*[-+0-9.eEdD]+ *eV"),
                SLParser(r"\s*\|\s*van der Waals energy corr\.\s*:\s*(?P<vanDerWaalsEnergyTS>[-+0-9.eEdD]+) *Ha\s*[-+0-9.eEdD]+ *eV", kind=MatchKind.ExpectFail),
                SLParser(r"\s*\|\s*Entropy correction\s*:\s*(?P<entropyCorrectionScf0>[-+0-9.eEdD]+) *Ha\s*[-+0-9.eEdD]+ *eV"),
                SLParser(r"(?m)\s*\|\s*---------------*\s*$",kind=MatchKind.Sequenced|MatchKind.Weak),
                SLParser(r"\s*\|\s*Total energy\s*:\s*(?P<totalDftEnergyScf0>[-+0-9.eEdD]+)\s*Ha\s*[-+0-9.eEdD]+ eV"),
                SLParser(r"\s*\|\s*Total energy, T -> 0\s*:\s*(?P<totalDftEnergyT0Scf0>[-+0-9.eEdD]+) Ha\s*[-+0-9.eEdD]+ *eV"),
                SLParser(r"\s*\|\s*Electronic free energy\s*:\s*(?P<eletronicFreeEnergyScf0>[-+0-9.eEdD]+) Ha\s*[-+0-9.eEdD]+ *eV", kind=MatchKind.ExpectFail)]),
            LineParser("derivedEnergiesScf0",
                startReStr=r"\s*Derived energy quantities:",
                subParsers=[
                SLParser(r"\s*\|\s*Kinetic energy\s*:\s*(?P<electronicKineticEnergyScf0>[-+0-9.eEdD]+) Ha\s*[-+0-9.eEdD]+ *eV"),
                SLParser(r"\s*\|\s*Electrostatic energy\s*:\s*(?P<electrostaticEnergyScf0>[-+0-9.eEdD]+) Ha\s*[-+0-9.eEdD]+ *eV"),
                SLParser(r"\s*\|\s*Energy correction for multipole"),
                SLParser(r"\s*\|\s*error in Hartree potential\s*:\s*(?P<hartreeEnergyErrorScf0>[-+0-9.eEdD]+) Ha\s*[-+0-9.eEdD]+ *eV"),
                SLParser(r"\s*\|\s*Sum of eigenvalues per atom\s*:\s*(?P<sumEigenvaluesPerAtomScf0>[-+0-9.eEdD]+) eV",
                        localProps={'sumEigenvaluesPerAtomScf0':PropertyTransformer('sumEigenvaluesPerAtomScf0',
                            lambda x: x*eV)}),
                SLParser(r"\s*\|\s*Total energy \(T->0\) per atom\s*:\s*(?P<totalEnergyT0PerAtomScf0>[-+0-9.eEdD]+) eV",
                        localProps={'totalEnergyT0PerAtomScf0':PropertyTransformer('totalEnergyT0PerAtomScf0',
                            lambda x: x*eV)}),
                SLParser(r"\s*\|\s*Electronic free energy per atom\s*:\s*(?P<electronicFreeEnergyPerAtomScf0>[-+0-9.eEdD]+) eV", kind=MatchKind.ExpectFail,
                        localProps={'electronicFreeEnergyPerAtomScf0':PropertyTransformer('electronicFreeEnergyPerAtomScf0',
                            lambda x: x*eV)})]),
            SectionOpener("ScfIteration", kind=MatchKind.Sequenced|MatchKind.Repeating|MatchKind.ExpectFail,
                startReStr=r"\s*Begin self-consistency iteration #\s*(?P<scfIterationNr>[0-9]+)",
                subParsers=[
                SLParser(r" *Date *: *(?P<startDateScfIter>[-.0-9/]+) *, *Time *: *(?P<startTimeScfIter>[-+0-9.EDed]+)"),
                LineParser("TotalEnergyScfIter",
                    startReStr=r"\s*Total energy components:",
                    subParsers=[
                    SLParser(r"\s*\|\s*Sum of eigenvalues\s*:\s*(?P<evSumScfIter>[-+0-9.eEdD]+) *Ha\s*[-+0-9.eEdD]+ *eV"),
                    SLParser(r"\s*\|\s*XC energy correction\s*:\s*(?P<xcEnergyScfIter>[-+0-9.eEdD]+) *Ha\s*[-+0-9.eEdD]+ eV"),
                    SLParser(r"\s*\|\s*XC potential correction\s*:\s*(?P<xcPotentialEnergyScfIter>[-+0-9.eEdD]+) *Ha\s*[-+0-9.eEdD]+ *eV"),
                    SLParser(r"\s*\|\s*Free-atom electrostatic energy\s*:\s*(?P<eStaticEnergyScfIter>[-+0-9.eEdD]+) *Ha\s*[-+0-9.eEdD]+ *eV"),
                    SLParser(r"\s*\|\s*Hartree energy correction\s*:\s*(?P<hartreeEnergyScfIter>[-+0-9.eEdD]+) *Ha\s*[-+0-9.eEdD]+ *eV"),
                    SLParser(r"\s*\|\s*Entropy correction\s*:\s*(?P<entropyCorrectionScfIter>[-+0-9.eEdD]+) *Ha\s*[-+0-9.eEdD]+ *eV"),
                    SLParser(r"(?m)\s*\|\s*---------------*\s*$", kind=MatchKind.Sequenced|MatchKind.Weak),
                    SLParser(r"\s*\|\s*Total energy\s*:\s*(?P<totalDftEnergyScfIter>[-+0-9.eEdD]+)\s*Ha\s*[-+0-9.eEdD]+ eV"),
                    SLParser(r"\s*\|\s*Total energy, T -> 0\s*:\s*(?P<totalDftEnergyT0ScfIter>[-+0-9.eEdD]+) Ha\s*[-+0-9.eEdD]+ *eV"),
                    SLParser(r"\s*\|\s*Electronic free energy\s*:\s*(?P<eletronicFreeEnergyScfIter>[-+0-9.eEdD]+) Ha\s*[-+0-9.eEdD]+ *eV", kind=MatchKind.ExpectFail)]),
                LineParser("derivedEnergiesScfIter",
                    startReStr=r"\s*Derived energy quantities:",
                    subParsers=[
                    SLParser(r"\s*\|\s*Kinetic energy\s*:\s*(?P<electronicKineticEnergyScfIter>[-+0-9.eEdD]+) Ha\s*[-+0-9.eEdD]+ *eV"),
                    SLParser(r"\s*\|\s*Electrostatic energy\s*:\s*(?P<electrostaticEnergyScfIter>[-+0-9.eEdD]+) Ha\s*[-+0-9.eEdD]+ *eV"),
                    SLParser(r"\s*\|\s*Energy correction for multipole"),
                    SLParser(r"\s*\|\s*error in Hartree potential\s*:\s*(?P<hartreeEnergyErrorScfIter>[-+0-9.eEdD]+) Ha\s*[-+0-9.eEdD]+ *eV"),
                    SLParser(r"\s*\|\s*Sum of eigenvalues per atom\s*:\s*(?P<sumEigenvaluesPerAtomScfIter>[-+0-9.eEdD]+) eV",
                        localProps={'sumEigenvaluesPerAtomScfIter':PropertyTransformer('sumEigenvaluesPerAtomScfIter',
                            lambda x: x*eV)}),
                    SLParser(r"\s*\|\s*Total energy \(T->0\) per atom\s*:\s*(?P<totalEnergyT0PerAtomScfIter>[-+0-9.eEdD]+) eV",
                        localProps={'totalEnergyT0PerAtomScfIter':PropertyTransformer('totalEnergyT0PerAtomScfIter',
                            lambda x: x*eV)}),
                    SLParser(r"\s*\|\s*Electronic free energy per atom\s*:\s*(?P<electronicFreeEnergyPerAtomScfIter>[-+0-9.eEdD]+) eV", kind=MatchKind.ExpectFail,
                        localProps={'electronicFreeEnergyPerAtomScfIter':PropertyTransformer('electronicFreeEnergyPerAtomScfIter',
                            lambda x: x*eV)})]),
                LineParser("scfAccuracy",
                    startReStr=r"\s*Self-consistency convergence accuracy:",
                    subParsers=[
                    SLParser(r"\s*\|\s*Change of charge density\s*:\s*(?P<chargeDensityChangeScfIter>[-+0-9.eEdD]+)", kind=MatchKind.ExpectFail),
                    SLParser(r"\s*\|\s*Change of sum of eigenvalues\s*:\s*(?P<sumEigChangerScfIter>[-+0-9.eEdD]+) *eV",
                        localProps={'sumEigChangerScfIter':PropertyTransformer('sumEigChangerScfIter',
                            lambda x: x*eV)}),
                    SLParser(r"\s*\|\s*Change of total energy\s*:\s*(?P<totalEnergyChangeScfIter>[-+0-9.eEdD]+) eV",
                        localProps={'totalEnergyChangeScfIter':PropertyTransformer('totalEnergyChangeScfIter',
                            lambda x: x*eV)}),
                    ])
                ])
            ]),
        LineParser("Hirshfeld",kind=MatchKind.Sequenced|MatchKind.ExpectFail,
            startReStr=r"\s*Evaluating non-empirical van der Waals correction (Tkatchenko/Scheffler 2009)\.",
            subParsers=[
                SLParser(r"\s*Performing Hirshfeld analysis of fragment charges and moments\."),
                SLParser(r"\s*---------*")
            ]),
        LineParser("convergedEnergy",
            startReStr=r"\s*Energy and forces in a compact form:",
            subParsers=[
            SLParser(r"\s*\|\s*Total energy uncorrected\s*:\s*(?P<totalDftEnergy>[-+0-9.eEdD]+) *eV",
                localProps={'totalDftEnergy':PropertyTransformer('totalDftEnergy', lambda x: x*eV)}),
            SLParser(r"\s*\|\s*Total energy corrected\s*:\s*(?P<totalDftEnergyT0>[-+0-9.eEdD]+) *eV",
                localProps={'totalDftEnergyT0':PropertyTransformer('totalDftEnergyT0',lambda x: x*eV)}),
            SLParser(r"\s*\|\s*Electronic free energy\s*:\s*(?P<dftElectronicFreeEnergy>[-+0-9.eEdD]+) *eV", kind=MatchKind.ExpectFail,
                localProps={'dftElectronicFreeEnergy':PropertyTransformer('dftElectronicFreeEnergy',
                    lambda x: x*eV)})
            ]),
        LineParser("xcDecomposition",
            kind=MatchKind.ExpectFail,
            startReStr=r"\s*Start decomposition of the XC Energy",
            subParsers=[
            SLParser(r"\s*----------*",kind=MatchKind.Sequenced|MatchKind.Weak),
            SLParser(r"\s*Hartree-Fock part\s*:\s*(?P<exactExchangeEnergy>[-+0-9.eEdD]+) *Ha *[-+0-9.eEdD]+ *eV"),
            SLParser(r"\s*X Energy\s*:\s*(?P<xEnergyDft>[-+0-9.eEdD]+) *Ha *[-+0-9.eEdD]+ *eV"),
            SLParser(r"\s*C Energy GGA\s*:\s*(?P<cEnergyDft>[-+0-9.eEdD]+) *Ha *[-+0-9.eEdD]+ *eV"),
            SLParser(r"\s*LDA X and C from self-consistent density"),
            SLParser(r"\s*X Energy LDA\s*:\s*(?P<xEnergyLda>[-+0-9.eEdD]+)\s*Ha\s*[-+0-9.eEdD]+\s*eV"),
            SLParser(r"\s*C Energy LDA\s*:\s*(?P<cEnergyLda>[-+0-9.eEdD]+)\s*Ha\s*[-+0-9.eEdD]+\s*eV"),
            SLParser(r"\s*----------*",kind=MatchKind.Sequenced|MatchKind.Weak),
            SLParser(r"\s*End decomposition of the XC Energy")
            ]),
        SLParser(r".*?GW( quasiparticle)? calculation starts (?P<has_GW>\.\.\.)", kind=MatchKind.Sequenced|MatchKind.ExpectFail),
        LineParser("Mp2Parser",
            kind=MatchKind.Sequenced|MatchKind.ExpectFail,
            startReStr=r"\s*\|\s*MP2 calculation starts *\.\.\.",
            subParsers=[
            SLParser(r"\s*----------*",kind=MatchKind.Sequenced|MatchKind.Weak),
            SLParser(r"\s*\|\s*MP2 calculation comes to stop *\.\.\.",
                subParsers=[
                SLParser(r"\s*\|\s*Total time for calculating MP2 correction\s*:\s*(?P<mp2Time>[-+0-9.eEdD]+) *s"),
                SLParser(r"\s*----------*",kind=MatchKind.Sequenced|MatchKind.Weak),
                SLParser(r"\s*----------*",kind=MatchKind.Sequenced|MatchKind.Weak),
                SLParser(r"\s*\|\s*HF Energy\s*:\s*(?P<hfEnergy>[-+0-9.eEdD]+)\s*Ha\s*[-+0-9.eEdD]+\s*eV"),
                SLParser(r"\s*\|\s*MP2 correction\s*:\s*(?P<mp2EnergyCorrection>[-+0-9.eEdD]+)\s*Ha\s*[-+0-9.eEdD]+\s*eV"),
                SLParser(r"\s*----------*",kind=MatchKind.Sequenced|MatchKind.Weak),
                SLParser(r"\s*\|\s*Total Energy \+ MP2 correction\s*:\s*(?P<mp2Energy>[-+0-9.eEdD]+)\s*Ha\s*[-+0-9.eEdD]+\s*eV")
                ])
            ]),
        LineParser("finalEnergies",
            kind=MatchKind.ExpectFail,
            startReStr=r"\s*Final output of selected total energy values:",
            subParsers=[
            SLParser(r"\s*\|\s*Total energy of the DFT / Hartree-Fock s.c.f. calculation\s*:\s*(?P<finalTotalDftEnergy>[-+0-9.eEdD]+)\s*eV",
                localProps={'finalTotalDftEnergy':PropertyTransformer('finalTotalDftEnergy', lambda x: x*eV)}),
            SLParser(r"\s*\|\s*Final zero-broadening corrected energy \(caution - metals only\)\s*:\s*(?P<finalTotalDftEnergyT0>[-+0-9.eEdD]+)\s*eV",
                localProps={'finalTotalDftEnergyT0':PropertyTransformer('finalTotalDftEnergyT0', lambda x: x*eV)}),
            SLParser(r"\s*\|\s*Total energy after the post-s\.c\.f\. correlation calculation\s*:\s*(?P<finalCorrectedEnergy>[-+0-9.eEdD]+)\s*eV",
                kind=MatchKind.Sequenced|MatchKind.ExpectFail,
                localProps={'finalCorrectedEnergy':PropertyTransformer('finalCorrectedEnergy',lambda x: x*eV)}),
            SLParser(r"\s*\|\s*For reference only, the value of 1 Hartree used in FHI-aims is\s*:\s*(?P<eVHaValue>[-+0-9.eEdD]+)\s*eV")
            ]),
        SLParser(r"(?i) *Have a nice day\.",kind=MatchKind.Sequenced|MatchKind.AlwaysActive)
        ]
    )
]

FhiAimsControlInParser.setupControlInParsers(
        controlInSubParser=controlInSubParser,
        controlInOutputs=controlInOutputs)

controlInOutputs.subParsers.append(
        BandParser("BandCalculation", kind=MatchKind.Sequenced|MatchKind.Repeating|MatchKind.ExpectFail,
    startReStr=r"\s*Plot band\s(?P<bandNr>[0-9]+)",
                subParsers=[
    SLParser(r"\s*\|\s*begin\s+(?P<startX>[-+0-9.eEdD]+)\s+(?P<startY>[-+0-9.eEdD]+)\s+(?P<startZ>[-+0-9.eEdD]+)"),
    SLParser(r"\s*\|\s*end\s+(?P<endX>[-+0-9.eEdD]*\+)\s+(?P<endY>[-+0-9.eEdD]*\+)\s+(?P<endZ>[-+0-9.eEdD]+)"),
    SLParser(r"\s*\|\s*number of points:\s*(?P<nPoints>[0-9]+)")
    ]))

#dumpParsers(controlInSubParser)
FhiAimsControlInParser.setupGeometryInParsers(
        geometryInSubParser=geometryInSubParser,
        geometryInOutputs=geometryInOutputs)
#dumpParsers(geometryInSubParser)

class FhiAimsParser:
    """parses an fhi aims output, and connected files"""

    def __init__(self, normalizer, path):
        self.normalizer=normalizer
        self.messages = []
        self.errorLevel=0
        self.logger=Logger(self.addMessage)
        self.aimsOutputMetaInfo=normalizer.metaOfPath(path)
        self.path=path
        self.nRuns=0
        self.runInfo={}
        self.runInfoProps=set()
        self.runTraj=None
        self.runTrajProps=None
        self.runDir=None
        self.configGroup="main"
        self.configIndex=0

    def addMessage(self, message, level=0):
        if self.errorLevel<level:
            self.errorLevel=level
        res={
            'kind':Logger.labelForLevel(level),
            'message':message
        }
        self.messages.append(res)

    def datasetForProperty(self, prop, shape, extraIndexes=None):
        self.runTrajProps.add(prop.name)
        configs=self.runTraj.require_group("configs")
        groupConfig=configs.require_group(self.configGroup)
        propGroup=groupConfig.require_group(prop.name)
        mshape=list(shape)
        mshape.insert(0,None)
        ashape=list(shape)
        ashape.insert(0,1)
        gIndex=None
        nIndexes=1
        if extraIndexes:
            nIndexes+=len(extraIndexes)
        if not "lastDataset" in propGroup.attrs.keys():
            propGroup.attrs["lastDataset"]=0
            gIndex=propGroup.create_dataset("groupIndexes",shape=(1,nIndexes+2),
                maxshape=(None,nIndexes+2),dtype=numpy.int64)
            gIndex[0,0]=self.configIndex
            gIndex[0,nIndexes]=0
            gIndex[0,nIndexes+1]=0
            dt=prop.dtype
            if dt is str:
                dt = h5py.special_dtype(vlen=str)
            dataset=propGroup.create_dataset("d0",
                shape=tuple(ashape),
                maxshape=tuple(mshape),
                dtype=dt)
            return (0,dataset)
        else:
            gIndex=propGroup["groupIndexes"]
            gIndexLen=gIndex.shape[0]
            nStoredIndexes=gIndex.shape[1]
            if nStoredIndexes<nIndexes:
                raise Exception("increasing the number of indexes not supported")
            lastDataset=propGroup.attrs["lastDataset"]
            dataset=propGroup["d"+str(lastDataset)]
            gIndex.resize((gIndexLen+1,nIndexes+2))
            gIndex[gIndexLen,0]=self.configIndex
            if extraIndexes:
                gIndex[gIndexLen,1:nIndexes]=extraIndexes
            if nStoredIndexes>nIndexes:
                gIndex[gIndexLen,nIndexes:nStoredIndexes]=0
            if gIndexLen>0:
                oldIndexes=gIndex[gIndexLen-1,0:nStoredIndexes]
                newIndexes=gIndex[gIndexLen,0:nStoredIndexes]
                for iIndex in range(nStoredIndexes):
                    if newIndexes[iIndex]>oldIndexes[iIndex]:
                        break
                    if newIndexes[iIndex]<oldIndexes[iIndex]:
                        raise Exception("decreasing indexes {0} after {1} for dataset for property {2}".
                                format(newIndexes, oldIndexes, prop.name))
            if dataset.shape[1:]==shape:
                gIndex[gIndexLen,nStoredIndexes]=lastDataset
                ashape[0]=dataset.shape[0]+1
                gIndex[gIndexLen,nStoredIndexes+1]=ashape[0]
                dataset.resize(tuple(ashape))
                return (ashape[0]-1,dataset)
            lastDataset+=1
            gIndex[gIndexLen,nStoredIndexes]=lastDataset
            gIndex[gIndexLen,nStoredIndexes+1]=0
            propGroup.attrs.lastDataset=lastDataset
            dataset=propGroup.create_dataset("d"+str(lastDataset),
                shape=tuple(ashape),
                maxshape=tuple(mshape),
                dtype=prop.dtype)
            return (0,dataset)



    def parseBand(self, path, trajFile):
        config = trajFile['configs']['main']
        path_only=os.path.split(path)
        #print "parsing bands at", path_only[0]
        #Get number of spin channels
        max_spin_channel = 0
        Nsegments = 0
        isegments = []
        faimsout = open(path, "r")
        lines = faimsout.readlines()
        for i, line in enumerate(lines):
            if "Spin treatment: No spin polarisation." in line or "Spin handling was not defined in control.in. Defaulting to unpolarized case." in line:
                max_spin_channel = 1
                #print "One spin channel."
            elif "Spin treatment: Spin density functional theory - collinear spins." in line:
                max_spin_channel = 2
                #print "Two spin channels."
            #Get number of band segments
            elif "Plot band " in line:
                words = line.split()
                isegments += [ int(words[2]) ]
                Nsegments = len(isegments)
        faimsout.close()
        n,bandSegments = self.datasetForProperty(Property.withName('band_segm_start_end'), (Nsegments,2,3))  # start and end k-point for each segment
        n,bandLabels   = self.datasetForProperty(Property.withName('band_segm_labels'), (Nsegments,2))  # start and end label for each segment
        labels = []
        bandstartnames = []
        bandendnames = []
        faimsout=open(path, "r")
        lines = faimsout.readlines()
        for i, line in enumerate(lines):
            if "output band" in line:
                if "k_grid" in lines[i-1]:
                    for iband in range(i,i+Nsegments):
                        wordsbn = lines[iband].split()
                        bandstartnames += [ wordsbn[9] ]
                        bandendnames += [ wordsbn[10] ]
                        bandSegments[n,iband-i,0,:] = map(float,wordsbn[2:5])
                        bandSegments[n,iband-i,1,:] = map(float,wordsbn[5:8])
                    bandLabels[n,:,0] = bandstartnames
                    bandLabels[n,:,1] = bandendnames
        # nKpoints -> adjust below!
        faimsout.close()
        #Get number of k-points per band segment and number of KS-eigenvalues, only open one band.out file. UPDATE FOR NUMBER OF KPOINTS!
        idxx = []
        nKPoints = 0  #Update this later
        nEVs = 0
        fbandname = path_only[0]+"/band%i%03i.out"%(1,1)
        fband = open(fbandname, "r")
        for line in fband:
            words = line.split()
            idxx += [ int(words[0]) ] #Update this later
            nKPoints = len(idxx)     #Update this later
            nEVs = (len(words)-4)/2
        #print "There are %i k-points per band segment."%nKPoints
        #print "There are %i energies for each k-point."%nEVs
        fband.close()
        i,kPoints = self.datasetForProperty(Property.withName('band_k_points'), (Nsegments, nKPoints, 3))
        i,occupations = self.datasetForProperty(Property.withName('band_occupations'), (Nsegments, max_spin_channel, nKPoints, nEVs))
        i,energies = self.datasetForProperty(Property.withName('band_energies'), (Nsegments, max_spin_channel, nKPoints, nEVs))
        for iband in range(0,Nsegments):
            for ispin in range(0,max_spin_channel):
                iEV = 0
                idx = []
                path_only=os.path.split(path)
                fbandname = path_only[0]+"/band%i%03i.out"%(ispin+1,iband+1)
                fband = open(fbandname, "r")
                for line in fband:
                    words = line.split()
                    idx += [ int(words[0]) ]
                    iEV = len(idx)-1
                    kPoints[i, iband, iEV, 0] = float(words[1])
                    kPoints[i, iband, iEV, 1] = float(words[2])
                    kPoints[i, iband, iEV, 2] = float(words[3])
                    occupations[i, iband, ispin, iEV, :] =  map(float,words[4::2])
                    energies[i, iband, ispin, iEV, :] =  map(float,words[5::2])
                fband.close()

    def endRun(self,endKind):
        latticeVectors=TrajUtils.maybeGet(self.runInfo,["SimulationParameters","GeometryInValues","geometryIn_lattice_vector"])
        if latticeVectors:
            iDataSet,cell=self.datasetForProperty(Property.withName("cell"),(3,3))
            for iVect,lVect in enumerate(latticeVectors):
                cell[iDataSet,iVect,0]=lVect['x']*angstrom
                cell[iDataSet,iVect,1]=lVect['y']*angstrom
                cell[iDataSet,iVect,2]=lVect['z']*angstrom
        if not self.runDir is None:
            nLog.info("Writing normalized run of %s", self.path)
            self.runInfo["cleanEnd"] = (endKind == CloseKind.Normal)
            self.parseBand(self.path,self.runTraj)
            for k,v in self.runInfoInfo.items():
                if type(v) is set:
                    if k in self.runTrajInfo:
                        v2=self.runTrajInfo[k]
                        if type(v2) is set:
                            v2.update(v)
                            v2=list(v2)
                            v2.sort()
                            self.runTrajInfo[k]=v2
                    v=list(v)
                    v.sort()
                    self.runInfoInfo[k]=v

            rInfo={
                'info':self.runInfoInfo,
                'data':self.runInfo
            }
            iKind=list(self.runInfoProps)
            iKind.sort()
            #self.normalizer.jsonDump(rInfo,os.path.join(self.runDir,"runDesc.info.json")) #to do
            dt = h5py.special_dtype(vlen=str)
            dataset=self.runTraj.create_dataset("runDesc",
                shape=(1,),
                maxshape=(None,),
                dtype=dt)
            runInfoStr=json.dumps(self.runInfo,sort_keys=True,indent=2,separators=(',', ': '),
                ensure_ascii=False,check_circular=False)
            if sys.version_info.major < 3:
                if type(runInfoStr) is unicode:
                    runInfoStr=runInfoStr.encode("utf_8")
            dataset[0]=runInfoStr
            dataset2=self.runTraj.create_dataset("info",
                shape=(1,),
                maxshape=(None,),
                dtype=dt)
            infoStr=json.dumps(self.runTrajInfo,sort_keys=True,indent=2,separators=(',', ': '),
                ensure_ascii=False,check_circular=False)
            if sys.version_info.major < 3:
                if type(infoStr) is unicode:
                    infoStr=infoStr.encode("utf_8")
            dataset2[0]=infoStr
            self.runTraj.close()

            #dirMeta = self.normalizer.dirMetaOfPath(self.runDir) # to do
            #index=dirMeta.get("index",{})
            #runDescInfo = index.get("runDesc.info.json", {})
            #runDescInfo["fileName"]="runDesc.info.json"
            #runDescInfo["fileType"]="text/x-nomad-info+json"
            #index["runDesc.info.json"]=runDescInfo
            #dirMeta["index"]=index
            #self.normalizer.updateDirMetaOfPath(self.runDir,dirMeta)

            self.runDir=None
            self.runTraj=None
            self.runTrajProps=None
            self.runInfo={}
            self.runInfoProps=set()

    def newRun(self):
        if self.runDir:
            self.endRun()
        self.runDir="." #self.normalizer.createDirectoryForPath(self.path,"run") # to do
        self.nRuns+=1
        if len(self.runInfo)!=0:
            self.logger.addWarning("FhiAimsParser.newRun discarding self.runInfo content: {0}".format(self.runInfo))
        self.runInfo={
            'type':'FhiAimsNormalizedRunInfoV1_0',
            'origin':os.path.relpath(self.path,self.runDir),
            'runNumber':self.nRuns
        }
        #outSha224=self.normalizer.sha224OfPath(self.path)
        self.runInfoInfo={
                'name':os.path.basename(self.path)+'_run{0:03}'.format(self.nRuns),
                'type':'FhiAimsNormalizedRunInfoV1_0',
                'uri': [],
                'has_info_of_kind':set(),
                'contained_in':set(),
                'contains':set(),
                'related_info':set(),
                #'derived_from':set([outSha224]),
                'derived_info':set(),
                }
        self.runInfoProps=self.runInfoInfo['has_info_of_kind']
        self.runTrajInfo=dict(self.runInfoInfo)
        self.runTrajProps=self.runTrajInfo['has_info_of_kind']
        self.runTraj=h5py.File(os.path.join(self.runDir,'traj.info.hdf5'),"w")
        self.runTraj.attrs['format']='Hdf5NomadInfoV1_0'
        for k,v in self.runInfo.items():
            self.runTraj.attrs[k]=v
        #dirMeta = self.normalizer.dirMetaOfPath(self.runDir) # to do
        #index=dirMeta.get("index",{})
        #trajInfo = index.get("traj.info.hdf5",{})
        #trajInfo["fileType"]="application/x-nomad-info+hdf5"
        #trajInfo["fileName"]="traj.info.hdf5"
        #index["traj.info.hdf5"]=trajInfo
        #dirMeta["index"]=index
        #self.normalizer.updateDirMetaOfPath(self.runDir,dirMeta)

    def parse(self):
        nLog.info("parsing: %s",self.path)
        kwds = dict()
        if sys.version_info.major>2:
            kwds["encoding"]="utf_8"
        aimsOutputFile = open(self.path, "r", **kwds)
        self.aimsOutputLineFile = LineFile(aimsOutputFile)
        self.rootParser = FileParser(lineFile=self.aimsOutputLineFile, context=self, firstParserContext=SectionContext(sections[0]), repeatsFirst=True, rootParser=self)
        self.rootParser.parse()

if __name__ == "__main__":
    fileToParse = sys.argv[1]
    parser=FhiAimsParser(None, fileToParse)
    parser.parse()
