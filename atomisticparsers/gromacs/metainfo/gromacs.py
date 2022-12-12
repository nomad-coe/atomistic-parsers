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
from nomad.datamodel.metainfo import simulation
from nomad.datamodel.metainfo import workflow


m_package = Package()


class x_gromacs_mdin_input_output_files(MCategory):
    '''
    Parameters of mdin belonging to x_gromacs_section_control_parameters.
    '''

    m_def = Category()


class x_gromacs_mdin_control_parameters(MCategory):
    '''
    Parameters of mdin belonging to x_gromacs_section_control_parameters.
    '''

    m_def = Category()


class x_gromacs_mdin_method(MCategory):
    '''
    Parameters of mdin belonging to section method.
    '''

    m_def = Category()


class x_gromacs_mdout_single_configuration_calculation(MCategory):
    '''
    Parameters of mdout belonging to section_single_configuration_calculation.
    '''

    m_def = Category()


class x_gromacs_mdout_method(MCategory):
    '''
    Parameters of mdin belonging to section method.
    '''

    m_def = Category()


class x_gromacs_mdout_run(MCategory):
    '''
    Parameters of mdin belonging to settings run.
    '''

    m_def = Category()


class x_gromacs_mdin_run(MCategory):
    '''
    Parameters of mdin belonging to settings run.
    '''

    m_def = Category()


class x_gromacs_section_input_output_files(MSection):
    '''
    Section to store input and output file names
    '''

    m_def = Section(validate=False,)

    x_gromacs_inout_file_topoltpr = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs input topology file.
        ''',)

    x_gromacs_inout_file_trajtrr = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs input trajectory file.
        ''',)

    x_gromacs_inout_file_trajcompxtc = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs input compressed trajectory file.
        ''',)

    x_gromacs_inout_file_statecpt = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs input coordinates and state file.
        ''',)

    x_gromacs_inout_file_confoutgro = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs output configuration file.
        ''',)

    x_gromacs_inout_file_eneredr = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs output energies file.
        ''',)


class x_gromacs_section_control_parameters(MSection):
    '''
    Section to store the input and output control parameters
    '''

    m_def = Section(validate=False,)

    x_gromacs_inout_control_gromacs_version = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_precision = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_memory_model = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_mpi_library = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_openmp_support = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_gpu_support = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_opencl_support = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_invsqrt_routine = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_simd_instructions = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_fft_library = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_rdtscp_usage = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_cxx11_compilation = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_tng_support = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_tracing_support = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_built_on = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_built_by = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_build_osarch = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_build_cpu_vendor = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_build_cpu_brand = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_build_cpu_family = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_build_cpu_features = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_c_compiler = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_c_compiler_flags = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_cxx_compiler = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_cxx_compiler_flags = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_boost_version = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_integrator = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_tinit = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_dt = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_nsteps = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_initstep = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_simulationpart = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_commmode = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_nstcomm = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_bdfric = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_ldseed = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_emtol = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_emstep = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_niter = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_fcstep = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_nstcgsteep = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_nbfgscorr = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_rtpi = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_nstxout = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_nstvout = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_nstfout = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_nstlog = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_nstcalcenergy = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_nstenergy = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_nstxoutcompressed = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_compressedxprecision = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_cutoffscheme = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_nstlist = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_nstype = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_pbc = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_periodicmolecules = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_verletbuffertolerance = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_rlist = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_rlistlong = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_nstcalclr = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_coulombtype = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_coulombmodifier = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_rcoulombswitch = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_rcoulomb = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_epsilonr = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_epsilonrf = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_vdwtype = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_vdwmodifier = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_rvdwswitch = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_rvdw = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_dispcorr = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_tableextension = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_fourierspacing = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_fouriernx = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_fourierny = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_fouriernz = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_pmeorder = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_ewaldrtol = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_ewaldrtollj = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_ljpmecombrule = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_ewaldgeometry = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_epsilonsurface = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_implicitsolvent = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_gbalgorithm = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_nstgbradii = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_rgbradii = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_gbepsilonsolvent = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_gbsaltconc = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_gbobcalpha = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_gbobcbeta = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_gbobcgamma = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_gbdielectricoffset = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_saalgorithm = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_sasurfacetension = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_tcoupl = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_nsttcouple = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_nhchainlength = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_printnosehooverchainvariables = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_pcoupl = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_pcoupltype = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_nstpcouple = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_taup = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_compressibility = Quantity(
        type=np.dtype(np.float64),
        shape=[3, 3],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_compressibility0 = Quantity(
        type=np.dtype(np.float64),
        shape=[3],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_compressibility1 = Quantity(
        type=np.dtype(np.float64),
        shape=[3],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_compressibility2 = Quantity(
        type=np.dtype(np.float64),
        shape=[3],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_refp = Quantity(
        type=np.dtype(np.float64),
        shape=[3, 3],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_refp0 = Quantity(
        type=np.dtype(np.float64),
        shape=[3],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_refp1 = Quantity(
        type=np.dtype(np.float64),
        shape=[3],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_refp2 = Quantity(
        type=np.dtype(np.float64),
        shape=[3],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_refcoordscaling = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_posrescom = Quantity(
        type=np.dtype(np.float64),
        shape=[3],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_posrescom0 = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_posrescom1 = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_posrescom2 = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_posrescomb = Quantity(
        type=np.dtype(np.float64),
        shape=[3],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_posrescomb0 = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_posrescomb1 = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_posrescomb2 = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_qmmm = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_qmconstraints = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_qmmmscheme = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_mmchargescalefactor = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_ngqm = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_constraintalgorithm = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_continuation = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_shakesor = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_shaketol = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_lincsorder = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_lincsiter = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_lincswarnangle = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_nwall = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_walltype = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_wallrlinpot = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_wallatomtype = Quantity(
        type=np.dtype(np.float64),
        shape=[2],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_wallatomtype0 = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_wallatomtype1 = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_walldensity = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_walldensity0 = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_walldensity1 = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_wallewaldzfac = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_pull = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_rotation = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_interactivemd = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_disre = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_disreweighting = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_disremixed = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_drfc = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_drtau = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_nstdisreout = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_orirefc = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_oriretau = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_nstorireout = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_freeenergy = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_cosacceleration = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_deform = Quantity(
        type=np.dtype(np.float64),
        shape=[3, 3],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_deform0 = Quantity(
        type=np.dtype(np.float64),
        shape=[3],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_deform1 = Quantity(
        type=np.dtype(np.float64),
        shape=[3],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_deform2 = Quantity(
        type=np.dtype(np.float64),
        shape=[3],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_simulatedtempering = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_ex = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_ext = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_ey = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_eyt = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_ez = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_ezt = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_swapcoords = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_adress = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_userint1 = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_userint2 = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_userint3 = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_userint4 = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_userreal1 = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_userreal2 = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_userreal3 = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_userreal4 = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_nrdf = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_reft = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_taut = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_annealing = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_annealingnpoints = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_acc = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_nfreeze = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_energygrpflags = Quantity(
        type=np.dtype(np.float64),
        shape=[3, 2],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_energygrpflags0 = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_energygrpflags1 = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)

    x_gromacs_inout_control_energygrpflags2 = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs running environment and control parameters.
        ''',)


class x_gromacs_section_atom_to_atom_type_ref(MSection):
    '''
    Section to store atom label to atom type definition list
    '''

    m_def = Section(validate=False,)

    x_gromacs_atom_to_atom_type_ref = Quantity(
        type=np.dtype(np.int64),
        shape=['number_of_atoms_per_type'],
        description='''
        Reference to the atoms of each atom type.
        ''',)


class x_gromacs_section_single_configuration_calculation(MSection):
    '''
    section for gathering values for MD steps
    '''

    m_def = Section(validate=False,)


class System(simulation.system.System):

    m_def = Section(validate=False, extends_base_section=True,)

    x_gromacs_atom_positions_image_index = Quantity(
        type=np.dtype(np.int32),
        shape=['number_of_atoms', 3],
        unit='dimensionless',
        description='''
        PBC image flag index.
        ''',)

    x_gromacs_atom_positions_scaled = Quantity(
        type=np.dtype(np.float64),
        shape=['number_of_atoms', 3],
        unit='dimensionless',
        description='''
        Position of the atoms in a scaled format [0, 1].
        ''',)

    x_gromacs_atom_positions_wrapped = Quantity(
        type=np.dtype(np.float64),
        shape=['number_of_atoms', 3],
        unit='meter',
        description='''
        Position of the atoms wrapped back to the periodic box.
        ''',)

    x_gromacs_lattice_lengths = Quantity(
        type=np.dtype(np.float64),
        shape=[3],
        description='''
        Lattice dimensions in a vector. Vector includes [a, b, c] lengths.
        ''')

    x_gromacs_lattice_angles = Quantity(
        type=np.dtype(np.float64),
        shape=[3],
        description='''
        Angles of lattice vectors. Vector includes [alpha, beta, gamma] in degrees.
        ''')

    x_gromacs_dummy = Quantity(
        type=str,
        shape=[],
        description='''
        dummy
        ''',)

    x_gromacs_mdin_finline = Quantity(
        type=str,
        shape=[],
        description='''
        finline in mdin
        ''',)

    x_gromacs_traj_timestep_store = Quantity(
        type=str,
        shape=[],
        description='''
        tmp
        ''',)

    x_gromacs_traj_number_of_atoms_store = Quantity(
        type=str,
        shape=[],
        description='''
        tmp
        ''',)

    x_gromacs_traj_box_bound_store = Quantity(
        type=str,
        shape=[],
        description='''
        tmp
        ''',)

    x_gromacs_traj_box_bounds_store = Quantity(
        type=str,
        shape=[],
        description='''
        tmp
        ''',)

    x_gromacs_traj_variables_store = Quantity(
        type=str,
        shape=[],
        description='''
        tmp
        ''',)

    x_gromacs_traj_atoms_store = Quantity(
        type=str,
        shape=[],
        description='''
        tmp
        ''',)


class MolecularDynamics(workflow.MolecularDynamics):

    m_def = Section(validate=False, extends_base_section=True,)

    x_gromacs_barostat_target_pressure = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='pascal',
        description='''
        MD barostat target pressure.
        ''')

    x_gromacs_barostat_tau = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='second',
        description='''
        MD barostat relaxation time.
        ''')

    x_gromacs_barostat_type = Quantity(
        type=str,
        shape=[],
        description='''
        MD barostat type, valid values are defined in the barostat_type wiki page.
        ''')

    x_gromacs_integrator_dt = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='second',
        description='''
        MD integration time step.
        ''')

    x_gromacs_integrator_type = Quantity(
        type=str,
        shape=[],
        description='''
        MD integrator type, valid values are defined in the integrator_type wiki page.
        ''')

    x_gromacs_periodicity_type = Quantity(
        type=str,
        shape=[],
        description='''
        Periodic boundary condition type in the sampling (non-PBC or PBC).
        ''')

    x_gromacs_langevin_gamma = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='second',
        description='''
        Langevin thermostat damping factor.
        ''')

    x_gromacs_number_of_steps_requested = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        Number of requested MD integration time steps.
        ''')

    x_gromacs_thermostat_level = Quantity(
        type=str,
        shape=[],
        description='''
        MD thermostat level (see wiki: single, multiple, regional).
        ''')

    x_gromacs_thermostat_target_temperature = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='kelvin',
        description='''
        MD thermostat target temperature.
        ''')

    x_gromacs_thermostat_tau = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='second',
        description='''
        MD thermostat relaxation time.
        ''')

    x_gromacs_thermostat_type = Quantity(
        type=str,
        shape=[],
        description='''
        MD thermostat type, valid values are defined in the thermostat_type wiki page.
        ''')


class AtomParameters(simulation.method.AtomParameters):

    m_def = Section(validate=False, extends_base_section=True,)

    x_gromacs_atom_name = Quantity(
        type=str,
        shape=[],
        description='''
        Atom name of an atom in topology definition.
        ''',)

    x_gromacs_atom_type = Quantity(
        type=str,
        shape=[],
        description='''
        Atom type of an atom in topology definition.
        ''',)

    x_gromacs_atom_element = Quantity(
        type=str,
        shape=[],
        description='''
        Atom type of an atom in topology definition.
        ''',)

    x_gromacs_atom_type_element = Quantity(
        type=str,
        shape=[],
        description='''
        Element symbol of an atom type.
        ''',)

    x_gromacs_atom_type_radius = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        van der Waals radius of an atom type.
        ''',)

    number_of_atoms_per_type = Quantity(
        type=int,
        shape=[],
        description='''
        Number of atoms involved in this type.
        ''',)

    x_gromacs_atom_resid = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''',)

    x_gromacs_atom_resname = Quantity(
        type=str,
        shape=[],
        description='''
        ''',)

    x_gromacs_atom_molnum = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''',)

    x_gromacs_atom_moltype = Quantity(
        type=str,
        shape=[],
        description='''
        ''',)


class Interaction(simulation.method.Interaction):

    m_def = Section(validate=False, extends_base_section=True,)

    x_gromacs_interaction_atom_to_atom_type_ref = Quantity(
        type=simulation.method.AtomParameters,
        shape=['number_of_atoms_per_interaction'],
        description='''
        Reference to the atom type of each interaction atoms.
        ''',)

    x_gromacs_number_of_defined_pair_interactions = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        Number of defined pair interactions (L-J pairs).
        ''',)

    x_gromacs_pair_interaction_atom_type_ref = Quantity(
        type=simulation.method.AtomParameters,
        shape=['x_gromacs_number_of_defined_pair_interactions', 'number_of_atoms_per_interaction'],
        description='''
        Reference to the atom type for pair interactions.
        ''',)

    x_gromacs_pair_interaction_parameters = Quantity(
        type=np.dtype(np.float64),
        shape=['x_gromacs_number_of_defined_pair_interactions', 2],
        description='''
        Pair interactions parameters.
        ''',)

    x_gromacs_molecule_interaction_atom_to_atom_type_ref = Quantity(
        type=simulation.method.AtomParameters,
        shape=['number_of_atoms_per_interaction'],
        description='''
        Reference to the atom type of each molecule interaction atoms.
        ''',)

    x_gromacs_number_of_defined_molecule_pair_interactions = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        Number of defined pair interactions within a molecule (L-J pairs).
        ''',)

    x_gromacs_pair_molecule_interaction_parameters = Quantity(
        type=np.dtype(np.float64),
        shape=['number_of_defined_molecule_pair_interactions', 2],
        description='''
        Molecule pair interactions parameters.
        ''',)

    x_gromacs_pair_molecule_interaction_to_atom_type_ref = Quantity(
        type=simulation.method.AtomParameters,
        shape=['x_gromacs_number_of_defined_pair_interactions', 'number_of_atoms_per_interaction'],
        description='''
        Reference to the atom type for pair interactions within a molecule.
        ''',)


class Run(simulation.run.Run):

    m_def = Section(validate=False, extends_base_section=True,)

    x_gromacs_program_version_date = Quantity(
        type=str,
        shape=[],
        description='''
        Program version date.
        ''',)

    x_gromacs_parallel_task_nr = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        Program task no.
        ''',)

    x_gromacs_number_of_tasks = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        Number of tasks in parallel program (MPI).
        ''',)

    x_gromacs_program_module_version = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs program module (gmx) version.
        ''',)

    x_gromacs_program_license = Quantity(
        type=str,
        shape=[],
        description='''
        Gromacs program license.
        ''',)

    x_gromacs_xlo_xhi = Quantity(
        type=str,
        shape=[],
        description='''
        test
        ''',)

    x_gromacs_data_file_store = Quantity(
        type=str,
        shape=[],
        description='''
        Filename of data file
        ''',)

    x_gromacs_program_working_path = Quantity(
        type=str,
        shape=[],
        description='''
        tmp
        ''',)

    x_gromacs_program_execution_host = Quantity(
        type=str,
        shape=[],
        description='''
        tmp
        ''',)

    x_gromacs_program_execution_path = Quantity(
        type=str,
        shape=[],
        description='''
        tmp
        ''',)

    x_gromacs_program_module = Quantity(
        type=str,
        shape=[],
        description='''
        tmp
        ''',)

    x_gromacs_program_execution_date = Quantity(
        type=str,
        shape=[],
        description='''
        tmp
        ''',)

    x_gromacs_program_execution_time = Quantity(
        type=str,
        shape=[],
        description='''
        tmp
        ''',)

    x_gromacs_mdin_header = Quantity(
        type=str,
        shape=[],
        description='''
        tmp
        ''',)

    x_gromacs_mdin_wt = Quantity(
        type=str,
        shape=[],
        description='''
        tmp
        ''',)

    x_gromacs_section_input_output_files = SubSection(
        sub_section=SectionProxy('x_gromacs_section_input_output_files'),
        repeats=False)

    x_gromacs_section_control_parameters = SubSection(
        sub_section=SectionProxy('x_gromacs_section_control_parameters'),
        repeats=False)


class Constraint(simulation.system.Constraint):

    m_def = Section(validate=False, extends_base_section=True,)

    x_gromacs_input_units_store = Quantity(
        type=str,
        shape=[],
        description='''
        It determines the units of all quantities specified in the input script and data
        file, as well as quantities output to the screen, log file, and dump files.
        ''',)

    x_gromacs_data_bond_types_store = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        store temporarly
        ''',)

    x_gromacs_data_bond_count_store = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        store temporarly
        ''',)

    x_gromacs_data_angle_count_store = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        store temporarly
        ''',)

    x_gromacs_data_atom_types_store = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        store temporarly
        ''',)

    x_gromacs_data_dihedral_count_store = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        store temporarly
        ''',)

    x_gromacs_data_angles_store = Quantity(
        type=str,
        shape=[],
        description='''
        store temporarly
        ''',)

    x_gromacs_data_angle_list_store = Quantity(
        type=str,
        shape=[],
        description='''
        tmp
        ''',)

    x_gromacs_data_bond_list_store = Quantity(
        type=str,
        shape=[],
        description='''
        tmp
        ''',)

    x_gromacs_data_dihedral_list_store = Quantity(
        type=str,
        shape=[],
        description='''
        tmp
        ''',)

    x_gromacs_data_dihedral_coeff_list_store = Quantity(
        type=str,
        shape=[],
        description='''
        tmp
        ''',)

    x_gromacs_masses_store = Quantity(
        type=str,
        shape=[],
        description='''
        tmp
        ''',)

    x_gromacs_data_topo_list_store = Quantity(
        type=str,
        shape=[],
        description='''
        tmp
        ''',)

    x_gromacs_section_atom_to_atom_type_ref = SubSection(
        sub_section=SectionProxy('x_gromacs_section_atom_to_atom_type_ref'),
        repeats=True,)


class Calculation(simulation.calculation.Calculation):

    m_def = Section(validate=False, extends_base_section=True,)

    x_gromacs_section_single_configuration_calculation = SubSection(
        sub_section=SectionProxy('x_gromacs_section_single_configuration_calculation'),
        repeats=True,)
