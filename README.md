[![DOI](https://zenodo.org/badge/461062665.svg)](https://doi.org/10.5281/zenodo.13851190)

This is a collection of the NOMAD parsers for the following atomistic codes.

1. [Amber](http://ambermd.org/)
2. [ASAP](https://wiki.fysik.dtu.dk/asap)
3. [DFTB+](http://www.dftbplus.org/)
4. [DL_POLY](https://www.scd.stfc.ac.uk/Pages/DL_POLY.aspx)
5. [GROMACS](http://www.gromacs.org/)
6. [GROMOS](http://www.gromos.net/)
7. [GULP](http://gulp.curtin.edu.au/gulp/)
8. [LAMMPS](https://lammps.sandia.gov/)
9. [libAtoms](http://libatoms.github.io/)
11. [NAMD](http://www.ks.uiuc.edu/Research/namd/)
12. [openKIM](https://openkim.org/)
13. [Tinker](https://dasher.wustl.edu/tinker/)

Each of the parsers will read the relevant input and output files and provide all information in
NOMAD's unified Metainfo based Archive format.

## Preparing code input and output file for uploading to NOMAD

NOMAD accepts `.zip` and `.tar.gz` archives as uploads. Each upload can contain arbitrary
files and directories. NOMAD will automatically try to choose the right parser for you files.
For each parser (i.e. for each supported code) there is one type of file that the respective
parser can recognize. We call these files `mainfiles` as they typically are the main
output file a code. For each `mainfile` that NOMAD discovers it will create an entry
in the database that users can search, view, and download. NOMAD will associate all files
in the same directory as files that also belong to that entry. Parsers
might also read information from these auxillary files. This way you can add more files
to an entry, even if the respective parser/code might not directly support it.

To create an upload with all calculations in a directory structure:

```
zip -r <upload-file>.zip <directory>/*
```

Go to the [NOMAD upload page](https://nomad-lab.eu/prod/rae/gui/uploads) to upload files
or find instructions about how to upload files from the command line.

## Using the parser

You can use NOMAD's parsers and normalizers locally on your computer. You need to install
NOMAD's pypi package:

```
pip install nomad-lab
```

To parse code input/output from the command line, you can use NOMAD's command line
interface (CLI) and print the processing results output to stdout:

```
nomad parse --show-archive <path-to-file>
```

To parse a file in Python, you can program something like this:
```python
import sys
from nomad.cli.parse import parse, normalize_all

# match and run the parser
archive = parse(sys.argv[1])
# run all normalizers
normalize_all(archive)

# get the 'main section' section_run as a metainfo object
section_run = archive.section_run[0]

# get the same data as JSON serializable Python dict
python_dict = section_run.m_to_dict()
```

## Developing the parser

Create a virtual environment to install the parser in development mode:

```
pip install virtualenv
virtualenv -p `which python3` .pyenv
source .pyenv/bin/activate
```

Install NOMAD's pypi package:

```
pip install nomad-lab
```

Clone the atomistic parsers project and install it in development mode:

```
git clone https://github.com/nomad-coe/atomistic-parsers.git atomistic-parsers
pip install -e "atomistic-parsers[dev]"
```

## How to cite this work
Ladines, A.N., Chang, T., Daelman, N., Fekete, A., Ghiringhelli, L.M., Himanen, L., Ilyas, A., Mohr, B. Pizarro, Rudzinski, J.F., & Scheidgen, M Atomistic Parsers [Computer software]. https://doi.org/10.5281/zenodo.13851190