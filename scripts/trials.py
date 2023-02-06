#Imports
import numpy as np
import scipy as sp
import healpy as hp
import histlite as hl
import csky as cy
import os
import gc
from glob import glob

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from modules.binnedtemplate import LiMaStats

import argparse

p = argparse.ArgumentParser(description='Trial creation for Li, Ma-based binned template analysis.',
                            formatter_class=argparse.RawTextHelpFormatter)

#Arguments for class object initiation:

p.add_argument('--data-path', type=str, required=True,
               help='Path to numpy array(s) (directory or single file) of data, or Path to numpy array of binned data (in this case, set --is-binned).')
p.add_argument('--sig-path', type=str, required=True, 
               help='Path to numpy array of simulation.')
p.add_argument('--grl-path', type=str, required=True, 
               help='Path to numpy array of good runs.')
p.add_argument('--is-binned', action='store_true',
               help='Use if data in data_path is already binned.')
p.add_argument('--savedir', type=str, default=None,
               help='Path to directory to save binned data if not --is-binned.')
p.add_argument('--name', type=str, default='LiMaBinnedTemplateAnalysis',
               help='Unique name for saving binned data.')
p.add_argument('--template-path', type=str, default=None,
               help='Path to template object/array.')
p.add_argument('--gamma', type=float, default=2.7,
               help='Spectral index for detector acceptance and injections (eventually).')
p.add_argument('--cutoff', type=float, default=10**(-1.0),
               help='Minimum template value to be considered an On bin.')
p.add_argument('--nside', type=int, default=128,
               help='Healpy NSIDE for binning.')
p.add_argument('--num-bands', type=int, default=100,
               help='Number of evenly spaced sin(Dec) bands for calculating likelihoods.')
p.add_argument('--min-dec-deg', type=float, default=-80,
               help='Minimum declination in degrees for likelihood calculations.')
p.add_argument('--max-dec-deg', type=float, default=80,
               help='Maximum declination in degrees for likelihood calculations.')
p.add_argument('--verbose', action='store_true',
               help='Use for more output.')
p.add_argument('--where-acc', type=str, default='both', choices=['both', 'temp', 'comb'],
               help='Where to include signal acceptance.')

#Arguments for getting trials:
p.add_argument('--num-trials', type=int, default=1,
               help='Number of trials to run.')
p.add_argument('--nsig', type=int, default=0,
               help='Number of signal events to inject.')
p.add_argument('--acc', type=str, default='signal', choices=['signal', 'bg'],
               help='Acceptance to use (should always be "signal").')
p.add_argument('--seed', type=int, default=None,
               help='Seed for RNG.')
p.add_argument('--save-trials', type=str, required=True, 
               help='Directory to save trials dictionary.')


args = p.parse_args()

#Initiate class object
lima_bean = LiMaStats(args.data_path, args.sig_path, args.grl_path, is_binned=args.is_binned, savedir=args.savedir, name=args.name, 
                     template=args.template_path, gamma=args.gamma, cutoff=args.cutoff, 
                     nside=args.nside, num_bands=args.num_bands, min_dec_deg=args.min_dec_deg, max_dec_deg=args.max_dec_deg, verbose=args.verbose, where_acc=args.where_acc)

#Run trials
trials = lima_bean.get_many_llrs(num=args.num_trials, n_sig=args.nsig, acc=args.acc, seed=args.seed, verbose=args.verbose)

#Save trials
cy.utils.ensure_dir(args.save_trials)
save_file = f"{args.save_trials}/trials{args.num_trials}_nsig{args.nsig}_seed{args.seed}_bothAcc_cutoff{args.cutoff}.npy"
np.save(save_file, trials)