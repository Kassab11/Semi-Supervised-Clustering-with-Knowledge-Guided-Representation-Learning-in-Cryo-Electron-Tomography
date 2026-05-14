# Semi-Supervised-Clustering-with-Knowledge-Guided-Representation-Learning-in-Cryo-Electron-Tomography
Semi-supervised clustering framework for cryo-electron tomography subtomograms using knowledge-guided representation learning, label-anchored GMM clustering, PCA voting, and confidence-based refinement for guided structural discovery.




dependincies:

```text
python 3.8
numpy
scipy
scikit-learn
scikit-image
matplotlib
pandas
tqdm
torch
torchvision
torchaudio
umap-learn
mrcfile
```

The uploaded environment uses CUDA 12.4 PyTorch builds:

```text
torch==2.4.0+cu124
torchvision==0.19.0+cu124
torchaudio==2.4.0+cu124
```


## Example command

Example for a 23-class simulated dataset with 1% labels from the known classes:

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


## Important arguments

| Argument | Description |
|---|---|
| `--training_data_path` | Pickle file containing subtomogram volumes. |
| `--labeled_indices_path` | Pickle file containing indices of labeled samples. |
| `--labeled_labels_path` | Pickle file containing labels for the labeled samples. |
| `--path_to_gt` | Optional full ground-truth labels for evaluation. |
| `--output_model_path` | Path where the final PyTorch model is saved. |
| `--output_label_path` | Path where final cluster labels are saved. |
| `--checkpoint_dir` | Directory where iteration checkpoints are saved. |
| `--outlogfile` | Text log file for per-iteration metrics. |
| `--candidateKs` | Candidate range for number of clusters, formatted as `minK,maxK`|
| `--true_k` | Ground-truth number of classes. Used only for aligned accuracy reporting. |
| `--M` | Number of outer iterations. |
| `--yopo_iteration` | Number of YOPO training epochs per outer iteration. |
| `--output_layer_iteration` | Number of epochs for training the classification head after clustering. |
| `--factor_use` | Data augmentation factor. Use `1` for no rotation augmentation. |
| `--normalize` | Whether to normalize subtomograms before training. |

## Outputs

After running, the script writes:

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


