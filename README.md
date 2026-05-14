# Semi-Supervised Clustering with Knowledge-Guided Representation Learning in Cryo-Electron Tomography

A semi-supervised clustering framework for cryo-electron tomography subtomograms.  
The framework combines knowledge-guided representation learning, label-anchored Gaussian Mixture Model clustering, PCA-based voting, and confidence-based refinement for guided structural discovery.

---

## Datasets

This project uses multiple cryo-ET datasets, including synthetic subtomograms generated with AITOM, processed Dragonfly CryoET data, synthetic tomograms generated with PolNet, and datasets from the CryoET Data Portal Challenge.

---

## Synthetic Subtomogram Data Generated with AITOM

To generate your own subtomograms from PDB files, see the [`cryoet-subtomogram-generation/`](./cryoet-subtomogram-generation/) directory.

The 23-class synthetic cryo-ET subtomogram dataset generated for this project is available on Hugging Face:

**Hugging Face Dataset:** [kassab11/cryo-et-subtomogram-generation](https://huggingface.co/datasets/kassab11/cryo-et-subtomogram-generation)

### Download

```bash
hf download kassab11/cryo-et-subtomogram-generation \
  --repo-type dataset \
  --local-dir Generated_Subtomograms
```

### Dataset Structure

```text
cryo-et-subtomogram-generation/
├── 0.1/
│   ├── 40/
│   └── 60/
├── 0.01/
│   ├── 40/
│   └── 60/
└── 0.03/
    ├── 40/
    └── 60/
```

The dataset is organized by signal-to-noise ratio, with SNR levels `0.1`, `0.01`, and `0.03`.

Each SNR folder contains two reconstruction conditions, `40/` and `60/`. Each condition includes:

- Generated density maps
- Subtomogram volumes
- Full ground-truth labels
- Labeled subset indices and labels used for semi-supervised learning

---

## Processed Dragonfly CryoET Dataset

The processed Dragonfly CryoET dataset used in this project is available on Hugging Face:

**Hugging Face Dataset:** [kassab11/Dragonfly_CryoET_dataset](https://huggingface.co/datasets/kassab11/Dragonfly_CryoET_dataset)

This dataset is based on the original Dryad dataset:

> Heebner, J.; Purnell, C.; Hylton, R. K.; Marsh, M.; Grillo, M. A.; Swulius, M. T.  
> *Deep learning training data (JOVE).* Dryad, 2022.  
> DOI: https://doi.org/10.5061/dryad.rxwdbrvct

The original cryo-ET data was processed, and corresponding ground-truth annotation masks were generated for segmentation and analysis workflows.

### Dataset Structure

```text
Dragonfly_CryoET_dataset/
├── tomogram/
│   └── processed tomogram files
└── annotation/
    └── generated ground-truth masks
```

---

## Synthetic Tomograms Generated with PolNet

The synthetic tomograms generated using PolNet are available on Hugging Face:

**Hugging Face Dataset:** [kassab11/synthetic-tomograms-using-polnet](https://huggingface.co/datasets/kassab11/synthetic-tomograms-using-polnet)

This dataset contains synthetic cryo-ET tomograms generated with PolNet for segmentation, detection, and analysis experiments.

### Download

```bash
hf download kassab11/synthetic-tomograms-using-polnet \
  --repo-type dataset \
  --local-dir "polnet_tomograms"
```

---

## CryoET Data Portal Challenge Dataset

This work also uses data from the **CZII CryoET Object Identification Challenge**, available through the **CryoET Data Portal**.

Relevant datasets:

| Dataset | Dataset ID |
|---|---|
| CZII - CryoET Object Identification Challenge - Experimental Training Data | `DS-10440` |
| CZII - CryoET Object Identification Challenge - Private Test Dataset | `DS-10446` |

Source: [CryoET Data Portal](https://cryoetdataportal.czscience.com/browse-data/datasets?search=challenge)

---

## Example Model Running Command

```bash
python model.py \
  --training_data_path "/path/to/subtomograms_v.pickle" \
  --labeled_indices_path "/path/to/subtomograms_labeled_indices.pickle" \
  --labeled_labels_path "/path/to/subtomograms_labeled_labels.pickle" \
  --path_to_gt "/path/to/subtomograms_all_labels_gt.pickle" \
  --output_model_path "/path/to/output/model.pth" \
  --output_label_path "/path/to/output/labels.pickle" \
  --checkpoint_dir "/path/to/output/checkpoints" \
  --outlogfile "/path/to/output/run_log.txt" \
  --candidateKs "20,25" \
  --true_k 23 \
  --M 20 \
  --yopo_iteration 10 \
  --output_layer_iteration 10 \
  --img_size 32 \
  --batch_size 32 \
  --lr 1e-4 \
  --factor_use 2 \
  --normalize True
```

---

## Important Arguments

| Argument | Description |
|---|---|
| `--training_data_path` | Path to the pickle file containing subtomogram volumes. |
| `--labeled_indices_path` | Path to the pickle file containing indices of labeled samples. |
| `--labeled_labels_path` | Path to the pickle file containing labels for the labeled samples. |
| `--path_to_gt` | Optional path to full ground-truth labels for evaluation. |
| `--output_model_path` | Path where the final trained PyTorch model is saved. |
| `--output_label_path` | Path where the final cluster labels are saved. |
| `--checkpoint_dir` | Directory where iteration checkpoints are saved. |
| `--outlogfile` | Text log file containing per-iteration metrics. |
| `--candidateKs` | Candidate range for the number of clusters, formatted as `minK,maxK`. |
| `--true_k` | Ground-truth number of classes. Used only for aligned accuracy reporting. |
| `--M` | Number of outer training iterations. |
| `--yopo_iteration` | Number of YOPO training epochs per outer iteration. |
| `--output_layer_iteration` | Number of epochs for training the classification head after clustering. |
| `--img_size` | Input subtomogram size. |
| `--batch_size` | Training batch size. |
| `--lr` | Learning rate. |
| `--factor_use` | Data augmentation factor. Use `1` for no rotation augmentation. |
| `--normalize` | Whether to normalize subtomograms before training. |

---

## Outputs

After running the model, the script writes the following outputs:

```text
model.pth                  # Final trained YOPO model
labels.pickle              # Final cluster assignments
checkpoints/checkpoint_*.pth
run_log.txt                # Per-iteration clustering metrics
```

If ground-truth labels are provided, the log reports:

```text
Homogeneity
Completeness
V-measure
Aligned clustering accuracy, when K matches true_k
```

---

## Notes

- Ground-truth labels are optional and are used only for evaluation.
- The `true_k` argument is used only for aligned accuracy reporting.
- The framework is designed for semi-supervised structural discovery in cryo-electron tomography subtomogram datasets.
