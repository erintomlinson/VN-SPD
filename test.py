import torch
import numpy as np
import sys
import os
import h5py
import copy
from pytorch3d.loss import chamfer_distance
from tqdm import tqdm

from data import create_dataset
from models import create_model
from options.test_options import TestOptions
from util import pc_utils


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
EPS = 1e-10

torch.manual_seed(0)
np.random.seed(seed=122222)


def load_transformation(transformation_path, device):
    fn = os.path.join(transformation_path)
    with h5py.File(fn, "r") as f:
        transform = np.asarray(f["transform"])

    return torch.FloatTensor(transform).to(device)

def main(opt):
    # hard-code some parameters for test
    opt.split = 'test'
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    model.eval()

    if opt.transform_path != '':
        presaved_trans = load_transformation(opt.transform_path)
    else:
        presaved_trans = None

    num_stability_exp = opt.num_stability_exp
    with torch.no_grad():

        # Full PC Consistency
        t_consistent_array = []
        rot_consistent_array = []
        rot_mats = []
        t_vecs = []
        for i, data in enumerate(tqdm(dataset, total=round(float(len(dataset)) / opt.batch_size))):
            model.set_input(data, use_rand_trans=False)  # unpack data from data loader

            out = model.test()  # run inference
            pc_at_canonic = out[0]
            rot_mat, t_vec = out[1]

            rot_mats.append(pc_utils.to_rotation_mat(rot_mat.detach(), opt.which_strict_rot))
            t_vecs.append(t_vec.view(-1, 3))

        rot_mats = torch.cat(rot_mats, dim=0)
        rot_mat_mean = pc_utils.to_rotation_mat(torch.mean(rot_mats, dim=0, keepdim=True), opt.which_strict_rot)
        rot_dists = pc_utils.cal_angular_metric(rot_mats, rot_mat_mean)
        rot_consistent_array.append(torch.sqrt(torch.mean(rot_dists ** 2)).item())

        t_vecs = torch.cat(t_vecs, dim=0)
        t_vec_mean = torch.mean(t_vecs, dim=0, keepdim=True)
        t_dists = torch.norm(t_vecs - t_vec_mean, dim=-1)
        t_consistent_array.append(torch.std(t_dists).item())

        print('')
        print("Full PC Rotation Consistency: ", np.mean(rot_consistent_array))
        print("Full PC Translation Consistency: ", np.mean(t_consistent_array))

        log_file = os.path.join(opt.checkpoints_dir, opt.name, 'full_canon_rot_dists.txt')
        np.savetxt(log_file, rot_dists.cpu().numpy())

        log_file = os.path.join(opt.checkpoints_dir, opt.name, 'full_canon_t_dists.txt')
        np.savetxt(log_file, t_dists.cpu().numpy())

        # Full PC Stability
        rot_stab_arr = []
        t_stab_arr = []
        for i, data in enumerate(tqdm(dataset, total=round(float(len(dataset))/opt.batch_size))):
            if (i *  opt.batch_size) > opt.num_test:
                break
            data_ = copy.deepcopy(data)
            rot_mats = []
            t_vecs = []
            for j in range(num_stability_exp):
                if not presaved_trans is None:
                    input_rot = presaved_trans[j][:3,:3].unsqueeze(0)
                    data_[0] = torch.bmm(data[0].transpose(1, 2).cuda(), input_rot).transpose(1, 2)
                    in_points, _ = model.set_input(data_, use_rand_trans=True)
                else:
                    in_points, trot = model.set_input(data_, use_rand_trans=True)
                    input_rot = trot.get_matrix()[:,:3,:3]
                    input_t_vec = trot.get_matrix()[:,:3,3].view(-1, 3)

                out = model.test()  # run inference
                pc_at_canonic = out[0]
                rot_mat, t_vec = out[1]

                rot_mats.append(torch.matmul(pc_utils.to_rotation_mat(rot_mat.detach(), opt.which_strict_rot), input_rot.transpose(1,2)))
                t_vecs.append(input_t_vec - t_vec.view(-1, 3))

            rot_mats = torch.cat(rot_mats, dim=0)
            rot_mat_mean = pc_utils.to_rotation_mat(torch.mean(rot_mats, dim=0, keepdim=True), opt.which_strict_rot)
            rot_dists = pc_utils.cal_angular_metric(rot_mats, rot_mat_mean)
            obj_rot_stability = torch.sqrt(torch.mean(rot_dists ** 2)).item()
            rot_stab_arr.append(obj_rot_stability)

            t_vecs = torch.cat(t_vecs, dim=0)
            t_vec_norms = torch.norm(t_vecs, dim=-1)
            obj_t_stability = torch.std(t_vecs).item()
            t_stab_arr.append(obj_t_stability)
            

        print('')
        print("Full PC Rotation Stability: ", np.mean(rot_stab_arr))
        print("Full PC Translation Stability: ", np.mean(t_stab_arr))

        num_partial_exp = opt.num_partial_exp
        if num_partial_exp > 0:
        
            # Partial PC Consistency
            t_consistent_array = []
            rot_consistent_array = []
            partial_rot_mats = []
            partial_t_vecs = []
            for i, data in enumerate(tqdm(dataset, total=round(float(len(dataset)) / opt.batch_size))):
                model.set_input(data, use_rand_trans=False)  # unpack data from data loader

                for j in range(num_partial_exp):

                    pc = in_points.clone().detach().cpu()
                    partial_pc, info = pc_utils.partialize_point_cloud(pc, prob=1.0, camera_direction='random')
                    partial_pc = partial_pc.cuda()

                    # run inference manually on partial_pc
                    inv_z, eq_z, partial_t_vec = model.netEncoder(partial_pc)
                    partial_rot_mat = model.netRotation(eq_z)
                    partial_pc_at_canonic = torch.matmul(partial_pc.permute(0, 2, 1) - partial_t_vec, partial_rot_mat.transpose(1, 2)).permute(0, 2, 1)
                    partial_t_vec = partial_t_vec.detach().view(-1, 3)

                    partial_rot_mats.append(pc_utils.to_rotation_mat(partial_rot_mat.detach(), opt.which_strict_rot))
                    partial_t_vecs.append(partial_t_vec)

            partial_rot_mats = torch.cat(partial_rot_mats, dim=0)
            partial_rot_mat_mean = pc_utils.to_rotation_mat(torch.mean(partial_rot_mats, dim=0, keepdim=True), opt.which_strict_rot)
            rot_dists = pc_utils.cal_angular_metric(partial_rot_mats, partial_rot_mat_mean)
            rot_consistent_array.append(torch.sqrt(torch.mean(rot_dists ** 2)).item())

            partial_t_vecs = torch.cat(partial_t_vecs, dim=0)
            partial_t_vec_mean = torch.mean(partial_t_vecs, dim=0, keepdim=True)
            t_dists = torch.norm(partial_t_vecs - partial_t_vec_mean, dim=-1)
            t_consistent_array.append(torch.std(t_dists).item())

            print('')
            print("Partial PC Rotation Consistency: ", np.mean(rot_consistent_array))
            print("Partial PC Translation Consistency: ", np.mean(t_consistent_array))

            log_file = os.path.join(opt.checkpoints_dir, opt.name, 'partial_canon_rot_dists.txt')
            np.savetxt(log_file, rot_dists.cpu().numpy())

            log_file = os.path.join(opt.checkpoints_dir, opt.name, 'partial_canon_t_dists.txt')
            np.savetxt(log_file, t_dists.cpu().numpy())

            # Partial PC Stability
            rot_stab_arr = []
            t_stab_arr = []
            partial_full_rot_dist_arr = []
            partial_full_t_dist_arr = []
            for i, data in enumerate(tqdm(dataset, total=round(float(len(dataset))/opt.batch_size))):
                if (i *  opt.batch_size) > opt.num_test:
                    break
                data_ = copy.deepcopy(data)
                partial_rot_mats = []
                partial_t_vecs = []
                partial_full_rot_dists = []
                partial_full_t_dists = []
                for j in range(num_stability_exp):

                    if not presaved_trans is None:
                        input_rot = presaved_trans[j][:3,:3].unsqueeze(0)
                        data_[0] = torch.bmm(data[0].transpose(1, 2).cuda(), input_rot).transpose(1, 2)
                        in_points, _ = model.set_input(data_, use_rand_trans=True)
                    else:
                        in_points, trot = model.set_input(data_, use_rand_trans=True)
                        input_rot = trot.get_matrix()[:,:3,:3]
                        input_t_vec = trot.get_matrix()[:,:3, 3].view(-1, 3)

                    out = model.test()  # run inference
                    pc_at_canonic = out[0]
                    rot_mat, t_vec = out[1]
                    t_vec = t_vec.detach().view(-1, 3)

                    for k in range(num_partial_exp):

                        pc = in_points.clone().detach().cpu()
                        partial_pc, info = pc_utils.partialize_point_cloud(pc, prob=1.0, camera_direction='random')
                        partial_pc = partial_pc.cuda()

                        # run inference manually on partial_pc
                        inv_z, eq_z, partial_t_vec = model.netEncoder(partial_pc)
                        partial_rot_mat = model.netRotation(eq_z)
                        partial_pc_at_canonic = torch.matmul(partial_pc.permute(0, 2, 1) - partial_t_vec, partial_rot_mat.transpose(1, 2)).permute(0, 2, 1)
                        partial_t_vec = partial_t_vec.detach().view(-1, 3)

                        partial_rot_mats.append(torch.matmul(pc_utils.to_rotation_mat(partial_rot_mat.detach(), opt.which_strict_rot), input_rot.transpose(1,2)))
                        partial_t_vecs.append(input_t_vec - partial_t_vec)

                        partial_full_rot_dist = pc_utils.cal_angular_metric(
                                pc_utils.to_rotation_mat(partial_rot_mat.detach(), opt.which_strict_rot), 
                                pc_utils.to_rotation_mat(rot_mat.detach(), opt.which_strict_rot))
                        partial_full_rot_dists.append(partial_full_rot_dist)

                        partial_full_t_dists.append(torch.norm(t_vec - partial_t_vec, dim=-1))


                partial_rot_mats = torch.cat(partial_rot_mats, dim=0)
                partial_rot_mat_mean = pc_utils.to_rotation_mat(torch.mean(partial_rot_mats, dim=0, keepdim=True), opt.which_strict_rot)
                rot_dists = pc_utils.cal_angular_metric(partial_rot_mats, partial_rot_mat_mean)
                obj_rot_stability = torch.sqrt(torch.mean(rot_dists ** 2)).item()
                rot_stab_arr.append(obj_rot_stability)
                
                partial_t_vecs = torch.cat(partial_t_vecs, dim=0)
                partial_t_vec_norms = torch.norm(partial_t_vecs, dim=-1)
                obj_t_stability = torch.std(partial_t_vecs).item()
                t_stab_arr.append(obj_t_stability)

                partial_full_rot_dists = torch.cat(partial_full_rot_dists, dim=0)
                partial_full_rot_dist_arr.append(torch.mean(partial_full_rot_dists).item())

                partial_full_t_dists = torch.cat(partial_full_t_dists, dim=0)
                partial_full_t_dist_arr.append(torch.mean(partial_full_t_dists).item())
            
            print('')
            print("Partial PC Rotation Stability: ", np.mean(rot_stab_arr))
            print("Partial PC Translation Stability: ", np.mean(t_stab_arr))
            print('')
            print("Avg Distance Between Partial and Full PC Pose: ", np.mean(partial_full_rot_dist_arr))
            print("Avg Distance Between Partial and Full PC Origin: ", np.mean(partial_full_t_dist_arr))

            log_file = os.path.join(opt.checkpoints_dir, opt.name, 'partial_full_rot_dists.txt')
            np.savetxt(log_file, np.array(partial_full_rot_dist_arr))

            log_file = os.path.join(opt.checkpoints_dir, opt.name, 'partial_full_t_dists.txt')
            np.savetxt(log_file, np.array(partial_full_t_dist_arr))
            

if __name__ == '__main__':
    args = TestOptions().parse()  # get training options
    main(args)
