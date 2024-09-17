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
from pydantic import Field
from typing import Optional

from nomad.config.models.plugins import ParserEntryPoint


class EntryPoint(ParserEntryPoint):
    parser_class_name: str = Field(
        description="""
        The fully qualified name of the Python class that implements the parser.
        This class must have a function `def parse(self, mainfile, archive, logger)`.
    """
    )
    code_name: Optional[str]
    code_homepage: Optional[str]
    code_category: Optional[str]
    metadata: Optional[dict] = Field(
        description="""
        Metadata passed to the UI. Deprecated. """
    )

    def load(self):
        from nomad.parsing import MatchingParserInterface

        return MatchingParserInterface(**self.dict())


amber_parser_entry_point = EntryPoint(
    name='parsers/amber',
    aliases=['parsers/amber'],
    description='NOMAD parser for AMBER.',
    python_package='atomisticparsers.amber',
    mainfile_contents_re=r'\s*Amber\s[0-9]+\s[A-Z]+\s*[0-9]+',
    parser_class_name='atomisticparsers.amber.AmberParser',
    code_name='Amber',
    code_homepage='http://ambermd.org/',
    code_category='Atomistic code',
    metadata={
        'codeCategory': 'Atomistic code',
        'codeLabel': 'Amber',
        'codeLabelStyle': 'only first character in capitals',
        'codeName': 'amber',
        'codeUrl': 'http://ambermd.org/',
        'parserDirName': 'dependencies/parsers/atomistic/atomisticparsers/amber/',
        'parserGitUrl': 'https://github.com/nomad-coe/atomistic-parsers.git',
        'parserSpecific': 'The Amber parser supports the SANDER and PMEMD Molecular Dynamics codes.',
        'status': 'production',
        'tableOfFiles': '',
    },
)

asap_parser_entry_point = EntryPoint(
    name='parsers/asap',
    aliases=['parsers/asap'],
    description='NOMAD parser for ASAP.',
    python_package='atomisticparsers.asap',
    mainfile_binary_header_re=b'AFFormatASE\\-Trajectory',
    mainfile_mime_re='application/octet-stream',
    mainfile_name_re=r'.*.traj$',  # can this be specified here? to directly check for the emt calculator maybe? somehow we need to seperate the general ase/traj parser from the asap parser
    parser_class_name='atomisticparsers.asap.AsapParser',
    code_name='ASAP',
    code_homepage='https://wiki.fysik.dtu.dk/asap',
    code_category='Atomistic code',
    metadata={
        'codeCategory': 'Atomistic code',
        'codeLabel': 'ASAP',
        'codeLabelStyle': 'all in capitals',
        'codeName': 'asap',
        'codeUrl': 'https://wiki.fysik.dtu.dk/asap',
        'parserDirName': 'dependencies/parsers/atomistic/atomisticparsers/asap/',
        'parserGitUrl': 'https://github.com/nomad-coe/atomistic-parsers.git',
        'parserSpecific': '',
        'preamble': '',
        'status': 'production',
        'tableOfFiles': '',
    },
)

ase_parser_entry_point = EntryPoint(
    name='parsers/ase',
    aliases=['parsers/ase'],
    description='NOMAD parser for ASE.',
    python_package='atomisticparsers.ase',
    mainfile_binary_header_re=b'.+?ASE\\-Trajectory',
    mainfile_mime_re='application/octet-stream',
    mainfile_name_re=r'.*.traj$',
    parser_class_name='atomisticparsers.ase.AseParser',
    code_name='ASE',
    code_homepage='https://wiki.fysik.dtu.dk/ase',
    code_category='Atomistic code',
    metadata={
        'codeCategory': 'Atomistic code',
        'codeLabel': 'ASE',
        'codeLabelStyle': 'all in capitals',
        'codeName': 'ase',
        'codeUrl': 'https://wiki.fysik.dtu.dk/ase',
        'parserDirName': 'dependencies/parsers/atomistic/atomisticparsers/ase/',
        'parserGitUrl': 'https://github.com/nomad-coe/atomistic-parsers.git',
        'parserSpecific': '',
        'preamble': '',
        'status': 'production',
        'tableOfFiles': '',
    },
)

bopfox_parser_entry_point = EntryPoint(
    name='parsers/bopfox',
    aliases=['parsers/bopfox'],
    description='NOMAD parser for BOPFOX.',
    python_package='atomisticparsers.bopfox',
    mainfile_contents_re=r'\-+\s+BOPfox \(v',
    parser_class_name='atomisticparsers.bopfox.BOPfoxParser',
    code_name='BOPfox',
    code_homepage='http://bopfox.de/',
    code_category='Atomistic code',
    metadata={
        'codeCategory': 'Atomistic code',
        'codeLabel': 'BOPfox',
        'codeLabelStyle': 'Capitals: B,O,P',
        'codeName': 'bopfox',
        'codeUrl': 'http://bopfox.de/',
        'parserDirName': 'dependencies/parsers/atomistic/atomisticparsers/bopfox/',
        'parserGitUrl': 'https://github.com/nomad-coe/atomistic-parsers.git',
        'parserSpecific': '',
        'preamble': '',
        'status': 'production',
        'tableOfFiles': '',
    },
)

dftbplus_parser_entry_point = EntryPoint(
    name='parsers/dftbplus',
    aliases=['parsers/dftbplus'],
    description='NOMAD parser for DFTBPLUS.',
    python_package='atomisticparsers.dftbplus',
    mainfile_contents_re=r'\|  DFTB\+',
    mainfile_mime_re='text/.*',
    parser_class_name='atomisticparsers.dftbplus.DFTBPlusParser',
    code_name='DFTB+',
    code_homepage='http://www.dftbplus.org/',
    code_category='Atomistic code',
    metadata={
        'codeCategory': 'Atomistic code',
        'codeLabel': 'DFTB+',
        'codeLabelStyle': 'all in capitals; use + instead of word',
        'codeName': 'dftbplus',
        'codeUrl': 'http://www.dftbplus.org/',
        'parserDirName': 'dependencies/parsers/atomistic/atomisticparsers/dftbplus/',
        'parserGitUrl': 'https://github.com/nomad-coe/atomistic-parsers.git',
        'parserSpecific': '',
        'preamble': '',
        'status': 'production',
        'tableOfFiles': '',
    },
)

dlpoly_parser_entry_point = EntryPoint(
    name='parsers/dl-poly',
    aliases=['parsers/dl-poly'],
    description='NOMAD parser for DLPOLY.',
    python_package='atomisticparsers.dlpoly',
    mainfile_contents_re=r'\*\*\s+DL_POLY.+\*\*',
    parser_class_name='atomisticparsers.dlpoly.DLPolyParser',
    code_name='DL_POLY',
    code_homepage='https://www.scd.stfc.ac.uk/Pages/DL_POLY.aspx',
    code_category='Atomistic code',
    metadata={
        'codeCategory': 'Atomistic code',
        'codeLabel': 'DL_POLY',
        'codeLabelStyle': 'all capitals and underscore, not dash (minus)',
        'codeName': 'dl-poly',
        'codeUrl': 'https://www.scd.stfc.ac.uk/Pages/DL_POLY.aspx',
        'parserDirName': 'dependencies/parsers/atomistic/atomisticparsers/dl-poly/',
        'parserGitUrl': 'https://github.com/nomad-coe/atomistic-parsers.git',
        'parserSpecific': '',
        'preamble': '',
        'status': 'production',
        'tableOfFiles': '',
    },
)

gromacs_parser_entry_point = EntryPoint(
    name='parsers/gromacs',
    aliases=['parsers/gromacs'],
    description='NOMAD parser for GROMACS.',
    python_package='atomisticparsers.gromacs',
    mainfile_contents_re=r'gmx mdrun, (VERSION|version)[\s\S]*Input Parameters:',
    parser_class_name='atomisticparsers.gromacs.GromacsParser',
    code_name='GROMACS',
    code_homepage='http://www.gromacs.org/',
    code_category='Atomistic code',
    metadata={
        'codeCategory': 'Atomistic code',
        'codeLabel': 'GROMACS',
        'codeLabelStyle': 'All in capitals',
        'codeName': 'gromacs',
        'codeUrl': 'http://www.gromacs.org/',
        'parserDirName': 'dependencies/parsers/atomistic/atomisticparsers/gromacs/',
        'parserGitUrl': 'https://github.com/nomad-coe/atomistic-parsers.git',
        'parserSpecific': '',
        'preamble': '',
        'status': 'production',
        'tableOfFiles': '',
    },
)

gromos_parser_entry_point = EntryPoint(
    name='parsers/gromos',
    aliases=['parsers/gromos'],
    description='NOMAD parser for GROMOS.',
    python_package='atomisticparsers.gromos',
    mainfile_contents_re=r'Bugreports to http://www.gromos.net',
    parser_class_name='atomisticparsers.gromos.GromosParser',
    code_name='GROMOS',
    code_homepage='http://www.gromos.net/',
    code_category='Atomistic code',
    metadata={
        'codeCategory': 'Atomistic code',
        'codeLabel': 'GROMOS',
        'codeLabelStyle': 'All in capitals',
        'codeName': 'gromos',
        'codeUrl': 'http://www.gromos.net/',
        'parserDirName': 'dependencies/parsers/atomistic/atomisticparsers/gromos/',
        'parserGitUrl': 'https://github.com/nomad-coe/atomistic-parsers.git',
        'parserSpecific': '',
        'preamble': '',
        'status': 'production',
        'tableOfFiles': '',
    },
)

gulp_parser_entry_point = EntryPoint(
    name='parsers/gulp',
    aliases=['parsers/gulp'],
    description='NOMAD parser for GULP.',
    python_package='atomisticparsers.gulp',
    mainfile_contents_re=(
        r'\s*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\s*\s*\*\s*GENERAL UTILITY '
        r'LATTICE PROGRAM\s*\*\s*'
    ),
    parser_class_name='atomisticparsers.gulp.GulpParser',
    code_name='GULP',
    code_homepage='http://gulp.curtin.edu.au/gulp/',
    code_category='Atomistic code',
    metadata={
        'codeCategory': 'Atomistic code',
        'codeLabel': 'GULP',
        'codeLabelStyle': 'All in capitals',
        'codeName': 'gulp',
        'codeUrl': 'http://gulp.curtin.edu.au/gulp/',
        'parserDirName': 'dependencies/parsers/atomistic/atomisticparsers/gulp/',
        'parserGitUrl': 'https://github.com/nomad-coe/atomistic-parsers.git',
        'parserSpecific': '',
        'preamble': '',
        'status': 'production',
        'tableOfFiles': '',
    },
)

h5md_parser_entry_point = EntryPoint(
    name='parsers/h5md',
    aliases=['parsers/h5md'],
    description='NOMAD parser for H5MD.',
    python_package='atomisticparsers.h5md',
    mainfile_binary_header_re=b'^\\x89HDF',
    mainfile_contents_dict={'__has_all_keys': ['h5md']},
    mainfile_mime_re='(application/x-hdf)',
    mainfile_name_re=r'^.*\.(h5|hdf5)$',
    parser_class_name='atomisticparsers.h5md.H5MDParser',
    code_name='H5MD',
    code_category='Atomistic code',
    metadata={
        'codeCategory': 'Atomistic code',
        'codeLabel': 'H5MD',
        'codeLabelStyle': 'All in capitals',
        'codeName': 'h5md',
        'parserDirName': 'dependencies/parsers/atomistic/atomisticparsers/h5md/',
        'parserGitUrl': 'https://github.com/nomad-coe/atomistic-parsers.git',
        'parserSpecific': '',
        'preamble': '',
        'status': 'production',
        'tableOfFiles': '',
    },
)

lammps_parser_entry_point = EntryPoint(
    name='parsers/lammps',
    aliases=['parsers/lammps'],
    description='NOMAD parser for LAMMPS.',
    python_package='atomisticparsers.lammps',
    mainfile_contents_re=r'^LAMMPS\s+\(.+\)',
    parser_class_name='atomisticparsers.lammps.LammpsParser',
    code_name='LAMMPS',
    code_homepage='https://lammps.sandia.gov/',
    code_category='Atomistic code',
    metadata={
        'codeCategory': 'Atomistic code',
        'codeLabel': 'LAMMPS',
        'codeLabelStyle': 'All in capitals',
        'codeName': 'lammps',
        'codeUrl': 'https://lammps.sandia.gov/',
        'parserDirName': 'dependencies/parsers/atomistic/atomisticparsers/lammps/',
        'parserGitUrl': 'https://github.com/nomad-coe/atomistic-parsers.git',
        'preamble': '',
        'status': 'production',
        'tableOfFiles': '',
    },
)

libatoms_parser_entry_point = EntryPoint(
    name='parsers/lib-atoms',
    aliases=['parsers/lib-atoms'],
    description='NOMAD parser for LIBATOMS.',
    python_package='atomisticparsers.libatoms',
    mainfile_contents_re=r'\s*<GAP_params\s',
    parser_class_name='atomisticparsers.libatoms.LibAtomsParser',
    code_name='libAtoms',
    code_homepage='http://libatoms.github.io/',
    code_category='Atomistic code',
    metadata={
        'codeCategory': 'Atomistic code',
        'codeLabel': 'libAtoms',
        'codeLabelStyle': 'Capitals: A',
        'codeName': 'lib-atoms',
        'codeUrl': 'http://libatoms.github.io/',
        'parserDirName': 'dependencies/parsers/atomistic/atomisticparsers/lib-atoms/',
        'parserGitUrl': 'https://github.com/nomad-coe/atomistic-parsers.git',
        'parserSpecific': '',
        'preamble': '',
        'status': 'production',
        'tableOfFiles': '',
    },
)

namd_parser_entry_point = EntryPoint(
    name='parsers/namd',
    aliases=['parsers/namd'],
    description='NOMAD parser for NAMD.',
    python_package='atomisticparsers.namd',
    mainfile_contents_re=r'\s*Info:\s*NAMD\s*[0-9.]+\s*for\s*',
    mainfile_mime_re='text/.*',
    parser_class_name='atomisticparsers.namd.NAMDParser',
    code_name='NAMD',
    code_homepage='http://www.ks.uiuc.edu/Research/namd/',
    code_category='Atomistic code',
    metadata={
        'codeCategory': 'Atomistic code',
        'codeLabel': 'NAMD',
        'codeLabelStyle': 'All in capitals',
        'codeName': 'namd',
        'codeUrl': 'http://www.ks.uiuc.edu/Research/namd/',
        'parserDirName': 'dependencies/parsers/atomistic/atomisticparsers/namd/',
        'parserGitUrl': 'https://github.com/nomad-coe/atomistic-parsers.git',
        'parserSpecific': '',
        'preamble': '',
        'status': 'production',
        'tableOfFiles': '',
    },
)

tinker_parser_entry_point = EntryPoint(
    name='parsers/tinker',
    aliases=['parsers/tinker'],
    description='NOMAD parser for TINKER.',
    python_package='atomisticparsers.tinker',
    mainfile_contents_re=r'TINKER  ---  Software Tools for Molecular Design',
    parser_class_name='atomisticparsers.tinker.TinkerParser',
    code_name='Tinker',
    code_homepage='https://dasher.wustl.edu/tinker/',
    code_category='Atomistic code',
    metadata={
        'codeCategory': 'Atomistic code',
        'codeLabel': 'Tinker',
        'codeLabelStyle': 'Capitals: T',
        'codeName': 'tinker',
        'codeUrl': 'https://dasher.wustl.edu/tinker/',
        'parserDirName': 'dependencies/parsers/atomistic/atomisticparsers/tinker/',
        'parserGitUrl': 'https://github.com/nomad-coe/atomistic-parsers.git',
        'parserSpecific': '',
        'preamble': '',
        'status': 'production',
        'tableOfFiles': '',
    },
)

xtb_parser_entry_point = EntryPoint(
    name='parsers/xtb',
    aliases=['parsers/xtb'],
    description='NOMAD parser for XTB.',
    python_package='atomisticparsers.xtb',
    mainfile_contents_re=r'x T B\s+\|\s+\|\s+=',
    parser_class_name='atomisticparsers.xtb.XTBParser',
    code_name='xTB',
    code_homepage='https://www.chemie.uni-bonn.de/pctc/mulliken-center/software/xtb/',
    code_category='Atomistic code',
    metadata={
        'codeCategory': 'Atomistic code',
        'codeLabel': 'xTB',
        'codeLabelStyle': 'Capitals: TB',
        'codeName': 'xtb',
        'codeUrl': 'https://www.chemie.uni-bonn.de/pctc/mulliken-center/software/xtb/',
        'parserDirName': 'dependencies/parsers/atomistic/atomisticparsers/gromacs//xtb/',
        'parserGitUrl': 'https://github.com/nomad-coe/atomistic-parsers.git',
        'parserSpecific': '',
        'preamble': '',
        'status': 'production',
        'tableOfFiles': '|Input Filename| Description|\n|--- | --- |\n|`*.out` | **Mainfile:** a plain text file w/ **user-defined** name|\n|`*.coord`|plain text; structure file|\n|`*.xyz`| plain text, structure file|\n|`*xtbopt.log`|plain text, trajectory file of geometry optimization|\n|`*xtb.trj`|plain text, trajectory of molecular dynamics|\n|`*xtbtopo.mol`|plain text, topology file|\n|`*xtbrestart`|binary file, restart file|\n|`charges` |plain text, output charges|\n',
    },
)
