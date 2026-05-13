# AITom-Based Subtomogram Generation 

Generate simulated cryo-electron tomography subtomograms from PDB structures and prepare pickle files for semi-supervised DISCA-style clustering experiments.

This folder contains modified data-generation utilities based on the AITom project. The workflow was adapted to directly generate reconstructed subtomograms from PDB structures and save the output as Python pickle files for downstream semi-supervised clustering.



## Contents

```text
scripts/
  generate_aitom_subtomograms.py             # Convert PDB files to simulated reconstructed subtomograms
  prepare_disca_semisupervised_inputs.py     # Convert subtomogram records into DISCA-ready pickle files
  reconstruction__simple_convolution.py      # Local reconstruction simulation helper adapted from AITom
  reconstruction__util.py                    # Local reconstruction utility helper adapted from AITom
NOTICE.md                                    # Attribution and modification notice
requirements.txt                             # Python dependencies used directly by the scripts

```

## Important directory requirement

Keep all files in `scripts/` together in the same directory.

`generate_aitom_subtomograms.py` imports the local module `reconstruction__simple_convolution.py`, and `reconstruction__simple_convolution.py` imports `reconstruction__util.py`. If these files are separated, Python will not find the local imports.



## Attribution

These scripts are based on and modified from the AITom project:

- AITom GitHub repository: https://github.com/xulabs/aitom
- AITom license: GPL-3.0

Main modification in this folder: the generation workflow was adapted to directly generate subtomograms and save them as pickle files. The output records have the form:

```python
{"id": pdb_id, "uuid": uuid_unique, "v": subtomogram_volume}
```


## Main dependencies

dependencies:
  - python=3.8
  - pip
  - pip:
      - numpy==1.20.0
      - scipy==1.10.1
      - mrcfile==1.5.4
      - pillow==10.4.0
      - matplotlib==3.7.5
      - scikit-learn==1.3.2
      - numba==0.58.1
      - pyyaml==6.0.3

`generate_aitom_subtomograms.py` relies on AITom modules for volume I/O, resizing, rotations, multiprocessing, missing wedge simulation, CTF simulation, and image saving. It also calls Situs `pdb2vol` to convert PDB structures into density maps.

## Installation notes

2. Install Situs 3.2 and confirm that `pdb2vol` is available.
3. Keep the four Python scripts in the same directory.
4. Install the Python packages listed in `requirements.txt`.



## Input data

Place your PDB files in a folder such as:

```text
pdbs/
  1bxn.pdb
  1qvr.pdb
  1s3x.pdb
  ...
```

Each PDB filename is used as the class ID. For example, `1bxn.pdb` becomes class ID `1bxn` in the generated pickle records.

## Step 1: Generate subtomograms from PDB files

All machine-specific paths are provided through command-line arguments. The required inputs are:

| Argument | What the user provides |
|---|---|
| `--PDB_software` | Path to the Situs `pdb2vol` executable |
| `--PDB_dir` | Directory containing `.pdb` files |
| `--output_dir_density` | Output pickle path for resized density maps |
| `--output_path_sub` | Output pickle path for generated subtomograms |

Example command:

```bash
python scripts/generate_aitom_subtomograms.py \
  --PDB_software /path/to/Situs_3.2/bin/pdb2vol \
  --PDB_dir ./pdbs \
  --output_dir_density ./outputs/density.pickle \
  --output_path_sub ./outputs/subtomograms.pickle \
  --voxel_spacing 10.0 \
  --resolution 40.0 \
  --DensityMapSize 32 \
  --missing_wedge 60 \
  --SNR 0.03 \
  --sample_num 400
```

Important optional parameters:

| Option | Meaning | Default |
|---|---|---|
| `--voxel_spacing` | Voxel spacing in Angstrom | `10.0` |
| `--resolution` | Target resolution in Angstrom | `40.0` |
| `--DensityMapSize` | Cubic size of resized density maps | `32` |
| `--missing_wedge` | Missing wedge angle | `40` |
| `--SNR` | Signal-to-noise ratio | `0.03` |
| `--Dz` | Defocus in micrometers | `-5` |
| `--pix_size` | Pixel size in nm | `1` |
| `--voltage` | Accelerating voltage in keV | `300` |
| `--Cs` | Spherical aberration in mm | `2.7` |
| `--loc` | Maximum random translation fraction | `0.1` |
| `--sample_num` | Number of subtomograms generated per PDB class | `400` |

Optional image export:

```bash
python scripts/generate_aitom_subtomograms.py \
  --PDB_software /path/to/Situs_3.2/bin/pdb2vol \
  --PDB_dir ./pdbs \
  --output_dir_density ./outputs/density.pickle \
  --output_path_sub ./outputs/subtomograms.pickle \
  --reconstructed_subtomogram_images \
  --reconstructed_subtomogram_images_dir ./outputs/images
```

## Step 2: Prepare semi-supervised DISCA pickle files

`prepare_disca_semisupervised_inputs.py` loads the generated `subtomograms.pickle`, converts string class IDs to integer labels, selects a labeled subset, and writes four pickle files.

All input/output paths are command-line arguments:

```bash
python scripts/prepare_disca_semisupervised_inputs.py \
  --input_pickle ./outputs/subtomograms.pickle \
  --output_dir ./outputs/prepared \
  --output_prefix subtomograms \
  --labeled_fraction 0.1 \
  --labeled_class_fraction 0.8 \
```

This writes:

```text
subtomograms_v.pickle                  # volumes, shape: (N, D, H, W, 1)
subtomograms_labeled_indices.pickle    # indices selected as labeled samples
subtomograms_labeled_labels.pickle     # labels for selected labeled samples
subtomograms_all_labels_gt.pickle      # full ground-truth label array
```


## Class ID mapping

The only user setting intentionally kept inside the script is `class_id_map` in `prepare_disca_semisupervised_inputs.py`.

By default it is empty:

```python
class_id_map = {}
```

When it is empty, labels are inferred automatically from the sorted unique IDs in the input pickle records. If you need a fixed label order, edit `class_id_map` manually. Example:

```python
class_id_map = {
    '1bxn': 0,
    '1qvr': 1,
    '1s3x': 2,
}
```

## Example used in the paper

For the AITom simulation example, the paper used 23 PDB-derived classes, including small/mid-size proteins such as `1BXN`, `1QVR`, `1S3X`, `1U6G`, `2CG9`, `3CF3`, `3D2F`, `3GL1`, `3H84`, and `3QM1`; larger complexes such as `2UV8`, `3IPM`, `3J9I`, `4CR2`, `4V4R`, `4V7R`, `4V94`, `5GJV`, `5MRC`, `6RD4`, and `6UTJ`; and the `30S` and `50S` ribosomal subunits extracted from `4V4R`.

The paper setup converted PDB structures into density maps using Situs 3.2, then used 10 Angstrom voxel spacing, 40 Angstrom target resolution, random 3D rotations/translations, missing wedge simulation, CTF modulation, Gaussian noise, and 400 subtomograms per class.

## Output format

The generated subtomogram pickle file contains a list of dictionaries:

```python
[
    {"id": "1bxn", "uuid": "...", "v": volume_array},
    {"id": "1qvr", "uuid": "...", "v": volume_array},
]
```

`prepare_disca_semisupervised_inputs.py` converts this list into arrays suitable for semi-supervised clustering.

## citation/acknowledgment 

If you use these scripts, please acknowledge AITom as the original source of the data-generation code and state that this folder contains modifications for direct subtomogram generation and pickle export.


