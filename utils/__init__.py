import yaml
import torch
import h5py as h5
import os
import datetime
import functools


def get_new_session_id():
    sess_id = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    if not os.path.exists('out'):
        os.makedirs('out')
    sess_dir = 'out/{}'.format(sess_id)
    if not os.path.exists(sess_dir):
        os.mkdir(sess_dir)
    return sess_id


def load_weights(model, path):
    state_dict = {}
    with h5.File(path, 'r') as file:
        for key, val in file.items():
            state_dict[key] = torch.from_numpy(val[...])
    model.load_state_dict(state_dict)


def store_weights(model, path):
    state_dict = model.state_dict()
    with h5.File(path, 'w') as f:
        for key, val in state_dict.items():
            f.create_dataset(key, data=val.numpy())


def timer(process_name):
    def decorator_timer(func):
        functools.wraps(func)
        def _wrapper(*args, **kwargs):
            tic = datetime.datetime.now()
            date_time = tic.strftime('%Y-%m-%d %H:%M:%S')
            print(f'{process_name} started at time {date_time}')
            func(*args, **kwargs)
            toc = datetime.datetime.now()
            date_time = toc.strftime('%Y-%m-%d %H:%M:%S')
            print(f'{process_name} ended at time {date_time}')
            print(f'{process_name} executed in {tic - toc}')
        return _wrapper
    return decorator_timer


def session(func):
    functools.wraps(func)
    def _wrapper(*args, **kwargs):
        sess_id = get_new_session_id()
        with open('out/{}/args.yaml'.format(sess_id), 'w') as args_file:
            yaml.dump(kwargs, args_file)
        func(session_id=sess_id, *args, **kwargs)
    return _wrapper
