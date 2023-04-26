import os
import sys
import torch
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import util.pc_utils as pc_utils
import plotly.express as px
colors = px.colors.qualitative.Vivid

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

def show_model_outputs(data, model, use_rand_trans=False, partialize=False, height=500, width=1200, showaxis=False):
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
    # fig.update_layout(margin=dict(r=0, l=0))
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
