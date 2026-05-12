# Attribution and License Notice

The Python scripts in `scripts/` are adapted from the AITom project:

- https://github.com/xulabs/aitom

The original AITom project is licensed under the GNU General Public License v3.0 (GPL-3.0).
This GitHub repository is licensed under GPL-3.0 at the repository root, so a separate license
inside this folder is not needed.

## Modifications

The included scripts were modified for this workflow to:

- generate reconstructed cryo-ET subtomograms directly from PDB-derived density maps,
- save generated subtomograms as Python pickle records,
- prepare DISCA-style pickle inputs for semi-supervised clustering,
- expose machine-specific paths and experiment settings through command-line arguments.

## Directory requirement

The scripts are intended to be kept together in the same directory. In particular,
`generate_aitom_subtomograms.py` imports `reconstruction__simple_convolution.py`, and
`reconstruction__simple_convolution.py` imports `reconstruction__util.py`.
