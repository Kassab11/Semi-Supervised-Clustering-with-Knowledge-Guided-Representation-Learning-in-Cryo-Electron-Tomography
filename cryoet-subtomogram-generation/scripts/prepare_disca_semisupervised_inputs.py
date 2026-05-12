import argparse
import os
import pickle

import numpy as np


# -----------------------------------------------------------------------------
# Class ID map
# -----------------------------------------------------------------------------
# Leave this dictionary empty to automatically assign integer labels from the
# sorted unique IDs found in the input pickle records.
#
# If you need a fixed class order, edit this dictionary manually. The keys must
# match the "id" values in the subtomogram records produced by generate_aitom_subtomograms.py.
# Example:
# class_id_map = {
#     '1bxn': 0,
#     '1qvr': 1,
#     '1s3x': 2,
# }
class_id_map = {}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare generated subtomogram pickle records for semi-supervised DISCA.'
    )

    parser.add_argument('--input_pickle', type=str, required=True,
                        help='Input subtomogram pickle file produced by generate_aitom_subtomograms.py')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory where prepared pickle files will be saved')
    parser.add_argument('--output_prefix', type=str, default='subtomograms',
                        help='Prefix for output pickle filenames. Default: subtomograms')
    parser.add_argument('--labeled_fraction', type=float, default=0.1,
                        help='Fraction of samples to label within each selected class. Default: 0.1')
    parser.add_argument('--labeled_class_fraction', type=float, default=1.0,
                        help='Fraction of classes that contain labeled samples. Default: 1.0')
    parser.add_argument('--random_seed', type=int, default=12,
                        help='Random seed for labeled subset selection. Default: 12')
    parser.add_argument('--add_channel_dim', action='store_true', default=True,
                        help='Add a trailing channel dimension so volumes have shape (N, D, H, W, 1). Default: enabled')
    parser.add_argument('--no_channel_dim', action='store_false', dest='add_channel_dim',
                        help='Disable adding the trailing channel dimension')

    return parser.parse_args()


def load_records(input_pickle):
    with open(input_pickle, 'rb') as f:
        records = pickle.load(f, encoding='latin1')
    return records


def build_class_id_map(records):
    if class_id_map:
        return class_id_map

    unique_ids = sorted({rec['id'] for rec in records})
    inferred_map = {class_id: idx for idx, class_id in enumerate(unique_ids)}
    print('No class_id_map was provided in the script. Using inferred mapping:')
    for class_id, idx in inferred_map.items():
        print(f'  {class_id}: {idx}')
    return inferred_map


def records_to_arrays(records, id_map, add_channel_dim=True):
    all_volumes = []
    all_labels = []

    for rec in records:
        rec_id = rec['id']
        rec_vol = rec['v']

        rec_vol = np.array(rec_vol)
        if rec_vol.ndim != 3:
            raise ValueError(f'Volume must be 3D, got shape = {rec_vol.shape}')

        if rec_id not in id_map:
            raise ValueError(f"Unknown ID '{rec_id}' not found in class_id_map")

        numeric_label = id_map[rec_id]
        all_volumes.append(rec_vol)
        all_labels.append(numeric_label)

    all_volumes = np.array(all_volumes)
    all_labels = np.array(all_labels)

    print('all_volumes shape:', all_volumes.shape)
    print('all_labels shape: ', all_labels.shape)

    if add_channel_dim:
        all_volumes = np.expand_dims(all_volumes, axis=-1)
        print('After expand_dims ->', all_volumes.shape)

    return all_volumes, all_labels


def select_labeled_subset(all_labels, labeled_fraction, labeled_class_fraction, random_seed):
    if not (0 < labeled_fraction <= 1):
        raise ValueError('--labeled_fraction must be in the range (0, 1]')
    if not (0 < labeled_class_fraction <= 1):
        raise ValueError('--labeled_class_fraction must be in the range (0, 1]')

    np.random.seed(random_seed)

    unique_classes = np.unique(all_labels)
    num_classes = len(unique_classes)

    num_labeled_classes = max(1, int(labeled_class_fraction * num_classes))
    labeled_classes = np.random.choice(unique_classes, size=num_labeled_classes, replace=False)

    labeled_indices = []
    for cls in labeled_classes:
        class_indices = np.where(all_labels == cls)[0]
        num_to_label = max(1, int(labeled_fraction * len(class_indices)))
        labeled_in_class = np.random.choice(class_indices, size=num_to_label, replace=False)
        labeled_indices.extend(labeled_in_class)

    labeled_indices = np.array(labeled_indices)
    labeled_labels = all_labels[labeled_indices]

    print(f'Selected {len(labeled_classes)} labeled classes out of {num_classes}')
    print(f'Number of labeled samples = {len(labeled_indices)}')
    print('Labeled indices:', labeled_indices[:10])
    print('Labeled labels:', np.unique(np.array(labeled_labels)))

    return labeled_indices, labeled_labels


def save_outputs(output_dir, output_prefix, all_volumes, labeled_indices, labeled_labels, all_labels):
    os.makedirs(output_dir, exist_ok=True)

    output_paths = {
        'volumes': os.path.join(output_dir, f'{output_prefix}_v.pickle'),
        'labeled_indices': os.path.join(output_dir, f'{output_prefix}_labeled_indices.pickle'),
        'labeled_labels': os.path.join(output_dir, f'{output_prefix}_labeled_labels.pickle'),
        'all_labels_gt': os.path.join(output_dir, f'{output_prefix}_all_labels_gt.pickle'),
    }

    with open(output_paths['volumes'], 'wb') as f:
        pickle.dump(all_volumes, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(output_paths['labeled_indices'], 'wb') as f:
        pickle.dump(labeled_indices, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(output_paths['labeled_labels'], 'wb') as f:
        pickle.dump(labeled_labels, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(output_paths['all_labels_gt'], 'wb') as f:
        pickle.dump(all_labels, f, protocol=pickle.HIGHEST_PROTOCOL)

    print('Saved files:')
    for key, path in output_paths.items():
        print(f'  {key}: {path}')


def main():
    args = parse_args()

    records = load_records(args.input_pickle)
    id_map = build_class_id_map(records)
    all_volumes, all_labels = records_to_arrays(records, id_map, add_channel_dim=args.add_channel_dim)
    labeled_indices, labeled_labels = select_labeled_subset(
        all_labels,
        args.labeled_fraction,
        args.labeled_class_fraction,
        args.random_seed,
    )
    save_outputs(
        args.output_dir,
        args.output_prefix,
        all_volumes,
        labeled_indices,
        labeled_labels,
        all_labels,
    )

    print('Done preparing data for semi-supervised DISCA.')


if __name__ == '__main__':
    main()
