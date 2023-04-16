import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from data import create_dataset as _create_dataset
from options.test_options import TestOptions

def create_dataset(dataroot='data/shapenet', batch_size=1, shuffle=False):
    '''
    TODO
    '''
    args = f'--dataroot {dataroot} --class_choice plane'
    opt = TestOptions.parse(args)

    opt.split = 'test'
    opt.num_threads = 0
    opt.batch_size = batch_size
    opt.serial_batches = not(shuffle)
    dataset = _create_dataset(opt)

    return dataset, opt

