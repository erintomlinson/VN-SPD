import numpy as np
import h5py

n = 50
class_choice = 'plane'
src_file = f'valid/{class_choice}.h5'
dst_file = f'demo/{class_choice}.h5'

with h5py.File(src_file, 'r') as src:
    with h5py.File(dst_file, 'w') as dst:
        src_inds = np.random.choice(len(src['pcd']['point']), size=n, replace=False)
        pcd = dst.create_group('pcd')
        pcd_point = pcd.create_group('point')
        pcd_normal = pcd.create_group('normal')
        for dst_ind, src_ind in enumerate(src_inds):
            print(f'{src_ind} -> {dst_ind}')
            pcd_point[str(dst_ind)] = np.asarray(src['pcd']['point'][str(src_ind)]).astype(np.float32)
            pcd_normal[str(dst_ind)] = np.asarray(src['pcd']['normal'][str(src_ind)]).astype(np.float32)
