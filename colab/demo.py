import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from data import create_dataset as _create_dataset
from models import create_model as _create_model
from options.test_options import TestOptions

def create_dataset(model='shape_pose', batch_size=1, shuffle=False, quiet=True):
    '''
    TODO
    '''
    args = f'--dataroot data/shapenet --class_choice plane --model {model}'
    if quiet: args += ' --quiet'
    opt = TestOptions().parse(args.split())

    opt.split = 'test'
    opt.num_threads = 0
    opt.batch_size = batch_size
    opt.serial_batches = not(shuffle)
    dataset = _create_dataset(opt)
    return dataset, opt

def create_model(name, opt):
    '''
    TODO
    '''
    opt.name = name
    model = _create_model(opt)
    model.setup(opt)
    mode.eval()
    return model


