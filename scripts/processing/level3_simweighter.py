import simweights

import h5py
import numpy as np
import argparse

from icecube import astro

p = argparse.ArgumentParser(description="HDF5 MC simweight-ing",
                            formatter_class=argparse.RawTextHelpFormatter)
p.add_argument("--input", type=str, nargs='+', required=True, 
               help="Input hdf5 files")
p.add_argument("--output", type=str, required=True, 
               help="Output numpy file")
p.add_argument('--nfiles', type=int, nargs='+', required=True, 
               help='Number of MC files included in each input hdf5 file')

p.add_argument("--time", type=float, default=None,
               help="Time to assign to events")
p.add_argument("--fix-leap", action='store_true', dest="fix_leap",
               help="Apply leap second bug fix")

args = p.parse_args()

for inp in args.input:
    if inp[-5:] != ".hdf5":
        raise ValueError("Input file path must end with .hdf5")
if len(args.input) != len(args.nfiles):
    raise ValueError("Number of input files must match number of nfiles values")
if args.output[-4:] != ".npy":
    raise ValueError("Output file path must end with .npy")

#Get individual weighters
weighters = []
hdffiles = []
for i in range(len(args.input)):
    hdffiles.append(h5py.File(args.input[i], "r"))
    weighter = simweights.NuGenWeighter(hdffiles[i], nfiles=args.nfiles[i])
    weighters.append(weighter)

#combing weighters
combined_weighter = sum(weighters)

#===================
# Create numpy array
#===================
assign_time = args.time
leap_dt = 1. / 86400.
def leap_runs(runs):
    r""" Check if runs are inside the range affected
    by leap second bug. Only for pass1 & pass2 data.
    See:
    https://drive.google.com/file/d/0B6TW2cWERhC6OFFCbWxsTHB1VzQ/view

    Parameters
    ----------
    runs : int, np.ndarray
        Run or array of runs

    Returns
    -------
    mask : bool, np.ndarray
        True for runs affected by leap second bug
    """
    return (120398 <= runs) & (runs <= 126377)

def angular_distance(lon1,lat1,lon2,lat2):
    """
    calculate the angular distince along the great circle
    on the surface of a shpere between the points
    (`lon1`,`lat1`) and (`lon2`,`lat2`)

    This function Works for equatorial coordinates
    with right ascension as longitude and declination
    as latitude. This function uses the Vincenty formula
    for calculating the distance.

    Parameters
    ----------
    lon1 : array_like
      longitude of first point in radians
    lat1 : array_like
      latitude of the first point in radians
    lon2 : array_like
      longitude of second point in radians
    lat2 : array_like
      latitude of the second point in radians

    """

    c1 = np.cos(lat1)
    c2 = np.cos(lat2)
    s1 = np.sin(lat1)
    s2 = np.sin(lat2)
    sd = np.sin(lon2-lon1)
    cd = np.cos(lon2-lon1)

    return np.arctan2(
        np.hypot(c2*sd,c1*s2-s1*c2*cd),
        s1*s2+c1*c2*cd
        )

def mask_data(data):
    r""" Mask out events with bad angular errors

    Parameters
    ----------
    data : np.ndarray
        Array with data events

    Returns
    -------
    data : np.ndarray
        Array with masked data events
    """
    mask = (data["angErr"] < np.radians(30)) #& (data['logE']>0) & (data['logE']<10)
    return data[mask]

#dtypes for Numpy array
exp_dtype = [('run', int), ('event', int), ('subevent', int),
             ('ra', float), ('dec', float),
             ('azi', float), ('zen', float), ('time', float),
             ('logE', float), ('angErr', float), ('qtot', float), ('qtot_wdc', float)]

mc_dtype = [('true_ra', float), ('true_dec', float),
            ('true_azi', float), ('true_zen', float),
            ('true_energy', float), ('true_angErr', float), ('oneweight', float)]

#Get columns from combined weighter
run = combined_weighter.get_column('I3EventHeader', 'Run')
event = combined_weighter.get_column('I3EventHeader', 'Event')
subevent = combined_weighter.get_column('I3EventHeader', 'SubEvent')

zen = combined_weighter.get_column('SplineMPE', 'zenith')
azi = combined_weighter.get_column('SplineMPE', 'azimuth')

if assign_time is None:
    time = combined_weighter.get_column('I3EventHeader', 'time_start_mjd')
    if fix_leap:
        time = time[leap_runs(run)] + leap_dt
else:
    time = np.full_like(run, assign_time)
    
logE = np.log10(combined_weighter.get_column('SplineMPEMuEXDifferential', 'energy'))
qtot = combined_weighter.get_column('SRTHVInIcePulses_Qtot', 'value')
qtot_wdc = combined_weighter.get_column('SRTHVInIcePulses_QtotWithDC', 'value')

angErr = np.sqrt(combined_weighter.get_column('MPEFitParaboloidFitParams', 'err1')**2 + combined_weighter.get_column('MPEFitParaboloidFitParams', 'err2')**2) / np.sqrt(2)

trueZen = combined_weighter.get_column('I3MCWeightDict', 'PrimaryNeutrinoZenith')
trueAzi = combined_weighter.get_column('I3MCWeightDict', 'PrimaryNeutrinoAzimuth')
trueE = combined_weighter.get_column('I3MCWeightDict', 'PrimaryNeutrinoEnergy')

# END for (frame)
print("Found {} p frames".format(len(run)))

#construct numpy array
dtype = exp_dtype + mc_dtype
a = np.empty(len(run), dtype)

print(len(run), len(zen), len(azi), len(logE), len(trueE))

a['run'] = run
a['event'] = event
a['subevent'] = subevent
a['zen'] = zen
a['azi'] = azi
a['time'] = time
a['logE'] = logE
a['ra'], a['dec'] = astro.dir_to_equa(a['zen'], a['azi'], a['time'])
a['angErr'] = angErr
a['qtot'] = qtot
a['qtot_wdc'] = qtot_wdc

a['true_zen'] = trueZen
a['true_azi'] = trueAzi
a['true_energy'] = trueE

a['true_ra'], a['true_dec'] = astro.dir_to_equa(
    a['true_zen'], a['true_azi'], a['time'])

a['oneweight'] = combined_weighter.get_weights(1) / 2.0 # !!! NOTE !!! Division by two for NuSources convention (nu+nubar)

a['true_angErr'] = angular_distance(a['true_ra'], a['true_dec'], a['ra'], a['dec'])

#Cuts
data = a
if data.size:
    data = mask_data(data)
    
print("Total events found in final array: {}".format(len(data)))

#Save
print("\nSaving to {}".format(args.output))
np.save(args.output, data)

#Close HDF5 Files
for f in hdffiles:
    f.close()
