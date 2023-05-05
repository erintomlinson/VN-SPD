import os
import re
import sys
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
import util.pc_utils as pc_utils
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from data import create_dataset as _create_dataset
from models import create_model as _create_model
from options.test_options import TestOptions

def create_dataset(model='shape_pose', batch_size=1, shuffle=False):
    '''
    TODO
    '''
    args = f'--demo --dataroot data/shapenet --class_choice plane --model {model}'
    opt = TestOptions().parse(args.split())

    opt.split = 'test'
    opt.num_threads = 0
    opt.batch_size = batch_size
    opt.serial_batches = not(shuffle)
    opt.remove_knn = 0
    opt.no_input_resample = 0
    dataset = _create_dataset(opt)
    return dataset, opt

def create_model(name, opt):
    '''
    TODO
    '''
    opt.name = name
    model = _create_model(opt)
    model.setup(opt)
    model.eval()
    return model

def show_model_outputs(data, model, use_rand_trans=False, partialize=False, height=500, width=1200, showaxis=False, remove_outlier=True):
    '''
    TODO
    '''
    pcs = data[0].clone()
    batch_size = len(pcs)
    pcs, info = pc_utils.partialize_point_cloud(pcs, prob=float(partialize), camera_direction='random')

    model.eval()
    with torch.no_grad():
        _, trot = model.set_input((pcs, ), use_rand_trans=use_rand_trans)
        model.forward()

    if remove_outlier:
        outlier = torch.tensor([1.0, -1.0, -1.0], device=model.device).view(1, 3, 1)
        outlier_idx = torch.argwhere(torch.all(torch.isclose(model.recon_pc_inv, outlier, atol=1e-2), dim=1).squeeze()).item()
        model.recon_pc_inv[:, :, outlier_idx] = model.recon_pc_inv[:, :, 0]
        transformed_outlier = (torch.matmul(outlier.permute(0, 2, 1), model.rot_mat) + model.t_vec).permute(0, 2, 1)
        transformed_outlier_idx = torch.argwhere(torch.all(torch.isclose(model.recon_pc, transformed_outlier, atol=1e-2), dim=1).squeeze()).item()
        model.recon_pc[:, :, transformed_outlier_idx] = model.recon_pc[:, :, 0]

    subplots = {
        'Input': model.pc,
        'Re-posed Input': model.pc_at_inv,
        'Pose-Invariant Recon.': model.recon_pc_inv,
        'Input Recon.': model.recon_pc}

    fig = make_subplots(
        rows=batch_size, cols=len(subplots),
        subplot_titles=[subplot_name for _ in range(batch_size) for subplot_name in subplots.keys()],
        specs=[[{'type': 'scene'} for _ in range(len(subplots))] for _ in range(batch_size)],
        vertical_spacing=0, horizontal_spacing=0)

    colors = px.colors.qualitative.Vivid
    for subplot_idx, (subplot_name, subplot_data) in enumerate(subplots.items()):
        for pc_idx in range(batch_size):
            x, y, z = subplot_data[pc_idx].cpu().numpy()
            trace = go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(size=2, opacity=0.8, color=colors[subplot_idx]))
            fig.add_trace(trace, row=pc_idx+1, col=subplot_idx+1)

    fig.update_layout(height=height*batch_size, width=width, showlegend=False)
    fig.update_scenes(aspectmode='data', xaxis=dict(visible=showaxis), yaxis=dict(visible=showaxis), zaxis=dict(visible=showaxis))
    fig.show()


def get_pc(dataset, idx):
    '''
    TODO
    '''
    pc, label = dataset.dataset[idx]
    return (torch.tensor(pc)[None], torch.tensor([label]))


def show_pc(data, idx=0):
    '''
    TODO
    '''
    pcs = data[0].cpu().numpy()
    fig = go.Figure([go.Scatter3d(
        x=pcs[idx][0],
        y=pcs[idx][1],
        z=pcs[idx][2],
        mode='markers',
        marker=dict(size=2, opacity=1))
    ])
    fig.update_scenes(aspectmode='data')
    fig.show()


def show_pc_and_partial(data, idx=0, camera_direction='random', height=500, width=1000, showaxis=False):
    '''
    TODO
    '''
    pcs = data[0].cpu().numpy()
    partial_pcs = pc_utils.partialize_point_cloud(data[0], prob=1, camera_direction=camera_direction)[0].cpu().numpy()
    fig = make_subplots(
        rows=1, cols=2, 
        subplot_titles=['Complete', 'Partial'],
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        horizontal_spacing=0
    )
    colors = px.colors.qualitative.Vivid
    fig.add_trace(go.Scatter3d(
        x=pcs[idx][0],
        y=pcs[idx][1],
        z=pcs[idx][2],
        mode='markers',
        marker=dict(size=2, opacity=1, color=colors[0])),
        row=1, col=1
    )
    fig.add_trace(go.Scatter3d(
        x=partial_pcs[idx][0],
        y=partial_pcs[idx][1],
        z=partial_pcs[idx][2],
        mode='markers',
        marker=dict(size=2, opacity=1, color=colors[1])),
        row=1, col=2
    )
    plot_max = torch.max(torch.tensor(pcs[idx].T), dim=0)[0]
    plot_min = torch.min(torch.tensor(pcs[idx].T), dim=0)[0]
    plot_range = torch.vstack([plot_min, plot_max]).T

    fig.update_scenes(aspectmode='data')
    fig.update_scenes(xaxis=dict(range=plot_range[0]), yaxis=dict(range=plot_range[1]), zaxis=dict(range=plot_range[2]))
    fig.update_scenes(xaxis=dict(visible=showaxis), yaxis=dict(visible=showaxis), zaxis=dict(visible=showaxis))
    fig.update_layout(height=height, width=width, showlegend=False)
    fig.show()


def read_train_log(model_name):
    '''
    TODO
    '''
    log_file = os.path.join('checkpoints', model_name, 'loss_log.txt')
    train_log = []
    with open(log_file) as f:
        for line in f.readlines():
            if 'epoch' not in line:
                continue
            entry = {k: float(v) for k, v in re.findall('([0-9A-Za-z_]+): ([\.0-9eE+-]+)', line)}
            train_log.append(entry)

    return pd.DataFrame(train_log)


def plot_train_log(model_name, losses=None, figsize=None, ylim=None):
    '''
    TODO
    '''
    train_log = read_train_log(model_name)
    train_log = train_log.groupby('epoch').mean()
    train_log = train_log.drop(columns=['iters', 'time', 'data'])
    fig, ax = plt.subplots(figsize=figsize)
    if losses is not None:
        train_log[losses].plot(ax=ax)
    else:
        train_log.plot(ax=ax)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')
    ax.set_ylim(ylim)
    ax.set_title(model_name)
    ax.grid(alpha=0.3)


def read_model_metrics(model_name):
    '''
    TODO
    '''
    metric_path = os.path.join('checkpoints', model_name)
    metric_files = [f for f in glob(os.path.join(metric_path, '*.txt')) if all(s not in f for s in ['log', 'opt'])]
    metrics = dict()
    for metric_file in metric_files:
        metric_name = metric_file.split('/')[-1][:-4]
        metrics[metric_name] = np.loadtxt(metric_file)
    
    return metrics


def plot_model_metrics(model_name, bins=25):
    '''
    TODO
    '''
    metrics = read_model_metrics(model_name)
    
    subplots = {'full_canon_rot_dists': {'title': 'Full/Canonical Pose Consistency', 'xlabel': 'Axis-Angle Distance (deg)'},
                'partial_full_rot_dists': {'title': 'Full/Partial Pose Consistency', 'xlabel': 'Axis-Angle Distance (deg)'},
                'partial_full_recon_loss_ratios': {'title': 'Partial Reconstruction Quality', 'xlabel': 'Partial CD / Full CD'}}

    colors = sns.color_palette()
    fig, axes = plt.subplots(1, len(subplots), figsize=(14, 3))
    for ax, (metric_name, params) in zip(axes.flatten(), subplots.items()):
        data = metrics[metric_name]
        ax.hist(data, bins=bins, color=colors[0])
        ax.axvline(np.mean(data), color=colors[1], label=f'mean: {np.mean(data):.1f}')
        ax.axvline(np.median(data), color='r', label=f'median: {np.median(data):.1f}')
        ax.set_xlabel(params['xlabel'])
        ax.set_ylabel('Count')
        ax.grid(alpha=0.3)
        ax.legend(loc='upper right')
        ax.set_title(params['title'])
