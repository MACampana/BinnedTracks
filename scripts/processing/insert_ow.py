import numpy as np
import argparse

p = argparse.ArgumentParser(description="Adds oneweight array to MC array",
                            formatter_class=argparse.RawTextHelpFormatter)
p.add_argument("--MC", type=str, required=True, help="MC array file")
p.add_argument("--OW", type=str, required=True, help="OneWeight array file")

args = p.parse_args()

if args.MC[-4:] != ".npy":
    raise ValueError("Input files must be numpy array (.npy)")
if args.OW[-4:] != ".npy":
    raise ValueError("Input files must be numpy array (.npy)")

mc = np.load(args.MC)
ow = np.load(args.OW)

dtype = mc.dtype.descr + ow.dtype.descr

new_arr = np.empty(mc.shape, dtype=dtype)

new_arr['run'] = mc['run']
new_arr['event'] = mc['event']
new_arr['subevent'] = mc['subevent']
new_arr['zen'] = mc['zen']
new_arr['azi'] = mc['azi']
new_arr['time'] = mc['time']
new_arr['logE'] = mc['logE']
new_arr['ra'] = mc['ra']
new_arr['dec'] = mc['dec']
new_arr['angErr'] = mc['angErr']
new_arr['true_zen'] = mc['true_en']
new_arr['true_azi'] = mc['true_azi']
new_arr['true_energy'] = mc['true_energy']
new_arr['true_ra'] = mc['true_ra']
new_arr['true_dec'] = mc['true_dec']
new_arr['true_angErr'] = mc['true_angErr']

new_arr['oneweight'] = ow['oneweight']

#Save
outfile = args.MC.replace('.npy', '_wOW.npy')
np.save(outfile, new_arr)
print(f'MC array from {args.MC} with oneweights from {args.OW} saved to {outfile}.')