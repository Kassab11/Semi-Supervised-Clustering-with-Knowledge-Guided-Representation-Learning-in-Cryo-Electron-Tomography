import argparse
import os
import pickle
import subprocess
import tempfile
import uuid

import aitom.geometry.ang_loc as GAL
import aitom.geometry.rotate as GR
import aitom.image.io as IIO
import aitom.image.vol.util as IVU
import aitom.io.file as TIF
import aitom.parallel.multiprocessing.util as TPMU
import numpy as np
import reconstruction__simple_convolution as TSRSC


def convert(op):
    [fh, out_fn] = tempfile.mkstemp(
        prefix='tmp-%s-%d-%d-' % (op['pdb_id'], op['spacing'], op['resolution']),
        suffix='.mrc'
    )
    os.close(fh)

    assert os.path.isfile(op['pdb_file'])
    cmd = [str(op['situs_pdb2vol_program']), op['pdb_file'], out_fn]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        universal_newlines=True,
    )

    print(2, file=proc.stdin)                 # Mass-weight atoms: 1 = No, 2 = Yes
    print(1, file=proc.stdin)                 # Select atoms by B-factor threshold: 1 = No, 2 = Yes
    print(op['spacing'], file=proc.stdin)     # Voxel spacing in Angstrom
    print(-op['resolution'], file=proc.stdin) # Target resolution as negative value
    print(1, file=proc.stdin)                 # Gaussian smoothing kernel
    print(1, file=proc.stdin)                 # Correct lattice interpolation smoothing effects
    print(1, file=proc.stdin)                 # Kernel amplitude scaling factor

    proc.communicate()

    op['map'] = TIF.read_mrc_data(out_fn).astype('float')
    os.remove(out_fn)

    return op


class PDB2SUB():
    def __init__(self, loc):
        self.loc = loc

    def batch_processing(self, op):
        extension = '.pdb'
        pdb_path = {}
        for root, sub_folders, files in os.walk(op['pdb_dir']):
            for file_t in files:
                if not file_t.endswith(extension):
                    continue

                pdb_id = file_t[:-4]

                assert (pdb_id + extension) == file_t
                assert pdb_id not in pdb_path

                pdb_path[pdb_id] = os.path.join(root, file_t)

        if 'pdb_id_selected' in op and op['pdb_id_selected']:
            pdb_path = {_: pdb_path[_] for _ in (set(pdb_path.keys()) & set(op['pdb_id_selected']))}

        import copy
        ts = {}
        for pdb_id in pdb_path:
            for spacing in op['spacing_s']:
                for resolution in op['resolution_s']:
                    op_t = copy.deepcopy(op)
                    op_t['pdb_id'] = pdb_id
                    op_t['pdb_file'] = pdb_path[pdb_id]

                    assert 'resolution' not in op_t
                    op_t['resolution'] = resolution
                    assert 'spacing' not in op_t
                    op_t['spacing'] = spacing

                    ts[uuid.uuid4()] = {'func': convert, 'kwargs': {'op': op_t}}

        cre_s = TPMU.run_batch(ts)

        re = {}
        for cre in cre_s:
            pdb_id = cre['result']['pdb_id']
            resolution = cre['result']['resolution']
            spacing = cre['result']['spacing']

            if pdb_id not in re:
                re[pdb_id] = {}

            if spacing not in re[pdb_id]:
                re[pdb_id][spacing] = {}

            assert resolution not in re[pdb_id][spacing]
            re[pdb_id][spacing][resolution] = cre['result']

        return re

    def generate_map(self, v, op, signal_variance=None, verbose=False):
        loc_max = np.array(v.shape, dtype=float) * self.loc
        angle = GAL.random_rotation_angle_zyz()
        loc_r = (np.random.random(3) - 0.5) * loc_max
        vr = GR.rotate(v, angle=angle, loc_r=loc_r, default_val=0.0)
        vb = TSRSC.do_reconstruction(v=vr, op=op, signal_variance=signal_variance, verbose=verbose)
        return vb

    def save_pickle(self, path, buffer):
        output_dir = os.path.dirname(os.path.abspath(path))
        os.makedirs(output_dir, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(buffer, f, protocol=-1)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate cryo-ET subtomograms from PDB files and save them as pickle records.'
    )

    # Required user paths
    parser.add_argument('--PDB_software', type=str, required=True,
                        help='Path to the Situs pdb2vol executable, for example /path/to/Situs_3.2/bin/pdb2vol')
    parser.add_argument('--PDB_dir', type=str, required=True,
                        help='Directory containing input .pdb files')
    parser.add_argument('--output_dir_density', type=str, required=True,
                        help='Output pickle path for resized density maps, for example outputs/density.pickle')
    parser.add_argument('--output_path_sub', type=str, required=True,
                        help='Output pickle path for generated subtomograms, for example outputs/subtomograms.pickle')

    # Density-map parameters
    parser.add_argument('--voxel_spacing', '-vs', type=float, default=10.0,
                        help='Voxel spacing in Angstrom. Default: 10.0')
    parser.add_argument('--resolution', type=float, default=40.0,
                        help='Target resolution in Angstrom. Default: 40.0')
    parser.add_argument('--DensityMapSize', type=int, default=32,
                        help='Resize each density map to this cubic size. Default: 32')

    # Cryo-ET simulation parameters
    parser.add_argument('--missing_wedge', type=int, default=40,
                        help='Missing wedge angle. Default: 40')
    parser.add_argument('--SNR', type=float, default=0.03,
                        help='Signal-to-noise ratio. Default: 0.03')
    parser.add_argument('--Dz', type=float, default=-5,
                        help='Defocus in micrometers; negative is underfocus. Default: -5')
    parser.add_argument('--pix_size', type=float, default=1,
                        help='Pixel size in nm. Default: 1')
    parser.add_argument('--voltage', type=int, default=300,
                        help='Accelerating voltage in keV. Default: 300')
    parser.add_argument('--Cs', type=float, default=2.7,
                        help='Spherical aberration in mm. Default: 2.7')
    parser.add_argument('--ctf_envelop', type=float, default=None,
                        help='Optional CTF envelope sigma in Nyquist units. Default: None')
    parser.add_argument('--loc', type=float, default=0.1,
                        help='Maximum random translation as a fraction of volume size. Default: 0.1')
    parser.add_argument('--signal_variance', type=float, default=None,
                        help='Optional signal variance in reconstructed subtomograms. Default: None')
    parser.add_argument('--sample_num', type=int, default=400,
                        help='Number of subtomograms generated per PDB class. Default: 400')

    # Optional image export
    parser.add_argument('--reconstructed_subtomogram_images', action='store_true',
                        help='Save PNG images of generated subtomograms when this flag is provided')
    parser.add_argument('--reconstructed_subtomogram_images_dir', type=str, default='outputs/images',
                        help='Output directory for reconstructed subtomogram PNG images. Default: outputs/images')

    return parser.parse_args()


def main():
    args = parse_args()

    op_density = {
        'situs_pdb2vol_program': args.PDB_software,
        'spacing_s': [args.voxel_spacing],
        'resolution_s': [args.resolution],
        'pdb_dir': args.PDB_dir,
    }

    op_sub = {
        'model': {'missing_wedge_angle': args.missing_wedge, 'SNR': args.SNR},
        'ctf': {
            'pix_size': args.pix_size,
            'Dz': args.Dz,
            'voltage': args.voltage,
            'Cs': args.Cs,
            'sigma': args.ctf_envelop,
        },
    }

    pdb2sub_generator = PDB2SUB(args.loc)

    re = pdb2sub_generator.batch_processing(op_density)
    vols = {_: re[_][args.voxel_spacing][args.resolution]['map'] for _ in re}
    vols = IVU.resize_center_batch_dict(vs=vols, size=args.DensityMapSize, cval=0.0)

    pdb2sub_generator.save_pickle(args.output_dir_density, vols)

    if args.reconstructed_subtomogram_images:
        os.makedirs(args.reconstructed_subtomogram_images_dir, exist_ok=True)

    buffer_v = []
    for pdbs_id in vols.keys():
        for i in range(args.sample_num):
            v = vols[f'{pdbs_id}']
            vb = pdb2sub_generator.generate_map(v, op_sub, signal_variance=args.signal_variance)
            uuid_unique = str(uuid.uuid4())
            if args.reconstructed_subtomogram_images:
                IIO.save_png(
                    IVU.cub_img(-vb)['im'],
                    f'{args.reconstructed_subtomogram_images_dir}/{pdbs_id}_{i}.png',
                )
            v_save = {'id': pdbs_id, 'uuid': uuid_unique, 'v': vb}
            buffer_v.append(v_save)

    ids = [_['id'] for _ in buffer_v]
    print(f'Subtomogram IDs {np.unique(ids)}')
    print(f'Subtomograms Generated : {len(ids)}')
    pdb2sub_generator.save_pickle(args.output_path_sub, buffer_v)


if __name__ == '__main__':
    main()
