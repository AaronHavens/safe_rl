import argparse
import tensorflow as tf
from mpi4py import MPI
from rl_algs.common import set_global_seeds, tf_util as U
import os.path as osp
import gym, logging
import numpy as np
from collections import deque
from gym import spaces
import misc_util
import sys
import shutil
import subprocess
import master
#from adversaries import adv_gen

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
#parser.add_argument('savename', type=str)
parser.add_argument('--filename', type=str)
parser.add_argument('--task', type=str)
parser.add_argument('--num_subs', type=int)
parser.add_argument('--augment', type=str2bool)
parser.add_argument('--pretrain', type=int)
#parser.add_argument('--macro_duration', type=int)
parser.add_argument('--num_rollouts', type=int)
parser.add_argument('--warmup_time', type=int)
parser.add_argument('--train_time', type=int)
#parser.add_argument('--force_subpolicy', type=int)
parser.add_argument('--save', type=str)
#parser.add_argument('-s', action='store_true')
parser.add_argument('--load', type=str)
args = parser.parse_args()

#RELPATH = osp.join(args.save)
#LOGDIR = osp.join('/root/results' if sys.platform.startswith('linux') else '/tmp', RELPATH)

# def callback(it):
#     if MPI.COMM_WORLD.Get_rank()==0:
#         if it % 5 == 0 and it > 3 and not replay:
#             fname = osp.join("savedir/", 'checkpoints', '%.5i'%it)
#             U.save_state(fname)
#     if it == 0 and args.load is not None:
#         fname = osp.join("./savedir/",'checkpoints', args.load)
#         U.load_state(fname)
#         saver = tf.train.import_meta_graph('./savedir/test_partial/mc30.meta')
#         saver.restore(U.get_session(),tf.train.latest_checkpoint('./savedir/test_partial'))
#         pass

def train():
    num_timesteps=1e9
    seed = 1401
    rank = MPI.COMM_WORLD.Get_rank()
    sess = U.single_threaded_session()
    sess.__enter__()
    workerseed = seed + 1000 * MPI.COMM_WORLD.Get_rank()
    rank = MPI.COMM_WORLD.Get_rank()
    #set_global_seeds(workerseed)

    # if rank != 0:
    #     logger.set_level(logger.DISABLED)
    # logger.log("rank %i" % MPI.COMM_WORLD.Get_rank())

    world_group = MPI.COMM_WORLD.Get_group()
    mygroup = rank % 10
    theta_group = world_group.Incl([x for x in range(MPI.COMM_WORLD.size) if (x % 10 == mygroup)])
    comm = MPI.COMM_WORLD.Create(theta_group)
    comm.Barrier()
    # comm = MPI.COMM_WORLD

    master.start(args=args, workerseed=workerseed, rank=rank, comm=comm)

def main():
    # if MPI.COMM_WORLD.Get_rank() == 0 and osp.exists(LOGDIR):
    #     shutil.rmtree(LOGDIR)
    MPI.COMM_WORLD.Barrier()
    # with logger.session(dir=LOGDIR):
    train()

if __name__ == '__main__':
    main()
