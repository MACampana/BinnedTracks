#Imports
import os
os.environ['OMP_NUM_THREADS'] = '3' #limits number of threads used (like by hp.smoothing)

import numpy as np
import csky as cy

from modules.BinnedTemplate_SigSub import BinnedTemplateAnalysis

import argparse

p = argparse.ArgumentParser(description='Trial creation for binned template analysis.',
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
p.add_argument('--nside', type=int, default=128,
               help='Healpy NSIDE for binning.')
p.add_argument('--min-dec-deg', type=float, default=-80,
               help='Minimum declination in degrees for likelihood calculations.')
p.add_argument('--max-dec-deg', type=float, default=80,
               help='Maximum declination in degrees for likelihood calculations.')
p.add_argument('--verbose', action='store_true',
               help='Use for more output.')
p.add_argument('--qtot', action='store_true',
               help='Use for binning in log10 qtot rather than energy.')
p.add_argument('--ebins', type=float, nargs='+', default=None,
               help='If scalar, number of bins; if sequence, bin edges for log10 energy (or qtot).')
p.add_argument('--force', action='store_true',
               help='Use to force binned_data to given nside.')
p.add_argument('--sigsub', action='store_true',
               help='Use to use signal subtraction.')

#Arguments for getting trials:
p.add_argument('--num-trials', type=int, default=1,
               help='Number of trials to run.')
p.add_argument('--nsig', type=int, default=0,
               help='Number of signal events to inject.')
p.add_argument('--seed', type=int, default=None,
               help='Seed for RNG.')
p.add_argument('--poisson', action='store_true',
               help='Use to inject according to a poisson around nsig.')
p.add_argument('--save-trials', type=str, required=True, 
               help='Directory to save trials dictionary.')


args = p.parse_args()

#Initiate class object
bin_chilln = BinnedTemplateAnalysis(args.data_path, args.sig_path, args.grl_path, is_binned=args.is_binned, 
                                    savedir=args.savedir, name=args.name, 
                                    template=args.template_path, gamma=args.gamma, ebins=args.ebins, qtot=args.qtot, 
                                    nside=args.nside, min_dec_deg=args.min_dec_deg, max_dec_deg=args.max_dec_deg, verbose=args.verbose, force=args.force, 
                                    sigsub=args.sigsub)

#Run trials
trials = bin_chilln.get_many_fits(num=args.num_trials, n_sig=args.nsig, seed=args.seed, verbose=args.verbose, poisson=args.poisson)

#Save trials
cy.utils.ensure_dir(args.save_trials)
save_file = f"{args.save_trials}/trials{args.num_trials}_nsig{args.nsig}_seed{args.seed}.npy"
np.save(save_file, trials)
