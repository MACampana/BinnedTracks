
import os
import argparse
import pickle
import h5py
import numpy as np

from icecube import astro, paraboloid
from icecube import dataio, icetray, dataclasses, recclasses, simclasses

exp_dtype = [('run', int), ('event', int), ('subevent', int),
             ('ra', float), ('dec', float),
             ('azi', float), ('zen', float), ('time', float),
             ('logE', float), ('angErr', float)]

mc_dtype = [('true_ra', float), ('true_dec', float),
            ('true_azi', float), ('true_zen', float),
            ('true_energy', float), ('true_angErr', float)]#, ('oneweight', float)]

grl_dtype = np.dtype([('run', float), 
                      ('start', np.float64), 
                      ('stop', np.float64), 
                      ('livetime', float), 
                      ('events', int)])

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

#def ow(ow, nfiles, neventsperfile):
#    return ow / nfiles / neventsperfile

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

def i3f2npy(path, assign_time, fix_leap=True, MC=False):#, nfiles=None, neventsperfile=None):
    r""" Read events from i3file and return numpy array

    Parameters
    ----------
    path : str
        Path to i3file
    assign_time : float
        Time to assign to events (MC only)
    fix_leap : bool
        Fix leap second bug when true (only for pass1 & pass2)

    Returns
    -------
    a : np.ndarray
        Events formatted as numpy array
    """
    run, event, subevent = [], [], []
    azi, zen, angErr = [], [], []
    time, logE = [], []
    trueAzi, trueZen, trueE = [], [], []
    #oneweight = []
    angErr = []

    i3f = dataio.I3File(path)
    counter = 0
    for frame in i3f:
        if frame.Stop != icetray.I3Frame.Physics:
            continue
            
        if frame['I3EventHeader'].sub_event_stream == 'NullSplit':
            continue
            
        if 'MPEFit' not in frame.keys():
            continue
        
        if 'MPEFitMuEX' not in frame.keys():
            continue
            
        if not frame['FilterMask']['MuonFilter_13'].condition_passed:
            continue
            
            
        counter +=1 

        run.append(frame['I3EventHeader'].run_id)
        event.append(frame['I3EventHeader'].event_id)
        subevent.append(frame['I3EventHeader'].sub_event_id)

        zen.append(frame['MPEFit'].dir.zenith)
        azi.append(frame['MPEFit'].dir.azimuth)
        if assign_time is None:
            time.append(frame['I3EventHeader'].start_time.mod_julian_day_double)
            if fix_leap and leap_runs(run[-1]):
                time[-1] += leap_dt
        else:
            time.append(assign_time)
        logE.append(np.log10(frame['MPEFitMuEX'].energy))
        
        angErr.append(np.sqrt(frame['MPEFitCramerRaoParams'].cramer_rao_theta**2 + frame['MPEFitCramerRaoParams'].cramer_rao_phi**2 * np.sin(frame['MPEFit'].dir.zenith)**2) / np.sqrt(2) )

        if MC:
            # MC fields
            trueZen.append(frame['I3MCWeightDict']['PrimaryNeutrinoZenith'])
            trueAzi.append(frame['I3MCWeightDict']['PrimaryNeutrinoAzimuth'])
            trueE.append(frame['I3MCWeightDict']['PrimaryNeutrinoEnergy'])
            
            #oneweight.append(frame['I3MCWeightDict']['OneWeight'])
            
    # END for (frame)
    print("Found {} p frames".format(counter))

    if not MC:
        dtype = exp_dtype
    else:
        dtype = exp_dtype + mc_dtype
    a = np.empty(len(run), dtype)

    # return empty array if path is empty
    if a.size == 0:
        i3f.close()
        return a

    a['run'] = run
    a['event'] = event
    a['subevent'] = subevent
    a['zen'] = zen
    a['azi'] = azi
    a['time'] = time
    a['logE'] = logE
    a['ra'], a['dec'] = astro.dir_to_equa(a['zen'], a['azi'], a['time'])
    a['angErr'] = angErr

    if MC:
        # MC fields
        a['true_zen'] = trueZen
        a['true_azi'] = trueAzi
        a['true_energy'] = trueE

        a['true_ra'], a['true_dec'] = astro.dir_to_equa(
            a['true_zen'], a['true_azi'], a['time'])

        #a['oneweight'] = ow(oneweight, nfiles, neventsperfile)
        
        a['true_angErr'] = angular_distance(a['true_ra'], a['true_dec'], a['ra'], a['dec'])

    i3f.close()
    return a

# END i3f2npy()

def hdf2npy(path, assign_time, fix_leap=True, MC=False):#, nfiles=None, neventsperfile=None):
    r""" Read events from hdf5 and return numpy array
    
    !! NOT TESTED !!

    Parameters
    ----------
    path : str
        Path to i3file
    assign_time : float
        Time to assign to events (MC only)
    fix_leap : bool
        Fix leap second bug when true (only for pass1 & pass2)

    Returns
    -------
    a : np.ndarray
        Events formatted as numpy array
    """
    hdf = h5py.File(path, 'r')

    run = hdf['I3EventHeader']['Run']
    event = hdf['I3EventHeader']['Event']
    subevent = hdf['I3EventHeader']['SubEvent']

    zen = hdf['MPEFit']['zenith']
    azi = hdf['MPEFit']['azimuth']
    if assign_time is None:
        time = hdf['I3EventHeader']['time_start_mjd']
        if fix_leap:
            time = time[leap_runs(run)] + leap_dt
    else:
        time = np.full_like(run, assign_time)
    logE = np.log10(hdf['MPEFitMuEX']['energy'])

    angErr = np.sqrt(hdf['MPEFitCramerRaoParams']['CramerRaoTheta']**2 + hdf['MPEFitCramerRaoParams']['CramerRaoPhi']**2 * np.sin(zen)**2) / np.sqrt(2)

    if MC:
        # MC fields
        trueZen = hdf['I3MCWeightDict']['PrimaryNeutrinoZenith']
        trueAzi = hdf['I3MCWeightDict']['PrimaryNeutrinoAzimuth']
        trueE = hdf['I3MCWeightDict']['PrimaryNeutrinoEnergy']
        
        #oneweight = hdf['SimWeights']['OneWeight']
            
    # END for (frame)
    print("Found {} p frames".format(len(run)))

    if not MC:
        dtype = exp_dtype
    else:
        dtype = exp_dtype + mc_dtype + [('oneweight', float)]
    a = np.empty(len(run), dtype)

    # return empty array if path is empty
    if a.size == 0:
        hdf.close()
        return a

    a['run'] = run
    a['event'] = event
    a['subevent'] = subevent
    a['zen'] = zen
    a['azi'] = azi
    a['time'] = time
    a['logE'] = logE
    a['ra'], a['dec'] = astro.dir_to_equa(a['zen'], a['azi'], a['time'])
    a['angErr'] = angErr

    if MC:
        # MC fields
        a['true_zen'] = trueZen
        a['true_azi'] = trueAzi
        a['true_energy'] = trueE

        a['true_ra'], a['true_dec'] = astro.dir_to_equa(
            a['true_zen'], a['true_azi'], a['time'])

        #a['oneweight'] = oneweight
        
        a['true_angErr'] = angular_distance(a['true_ra'], a['true_dec'], a['ra'], a['dec'])
        
    hdf.close()
    return a

def mask_exp(data):
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
    mask = (data["angErr"] < np.radians(30))
    return data[mask]

def mask_mc(mc):
    r""" Mask out events with bad logE

    Parameters
    ----------
    MC : np.ndarray
        Array with MC events

    Returns
    -------
    data : np.ndarray
        Array with masked MC events
    """
    mask = (mc['logE'] > 0) 
    return mc[mask]

def proc(f, time, fix_leap=True, MC=False, hdf=False):#, nfiles=None, neventsperfile=None):
    r""" Perform i3 -> npy

    Parameters
    ----------
    f : str
        Path to hdf5 or i3 file with events
    time : float
        Time to assign to events (MC only)
    fix_leap : bool
        Fix leap second bug when true (only for pass1 & pass2)

    Returns
    -------
    data : np.ndarray
       Array with data events
    """
    # load data file
    data = []
    for filename in f:
        print("Extracting from ", filename)
        if hdf:
            x = hdf2npy(filename, time, fix_leap, MC)
        else:
            x = i3f2npy(filename, time, fix_leap, MC)#, nfiles, neventsperfile)

        if len(data) == 0: 
            data = x
        else: 
            data = np.concatenate([data, x])

    if data.size:
        data = mask_exp(data)

        if MC:
            # apply additional mask to Monte Carlo events
            data = mask_mc(data)

    return data

###############################################################################

########
# ARGS #
########

p = argparse.ArgumentParser(description="Performs i3 -> npy for L2.",
                            formatter_class=argparse.RawTextHelpFormatter)
p.add_argument("--input", type=str, nargs='+', required=True, help="Input I3 file(s)")
p.add_argument("--output", type=str, required=True, help="Output numpy file")
p.add_argument("--fix-leap", action='store_true', dest="fix_leap",
               help="Apply leap second bug fix")
p.add_argument("--hdf", action='store_true', dest="hdf",
               help="HDF5 --> npy instead of i3 --> npy (NOT TESTED)")

#For Monte Carlo only...
p.add_argument("--MC", action='store_true', dest="MC",
               help="Is Monte Carlo?")
p.add_argument("--time", type=float, default=None,
               help="Time to assign to events (MC only)")
#p.add_argument("--ow-file", type=str, default=None, 
#               help="File with saved oneweights (MC only)")
#p.add_argument("--nfiles", type=float, default=None,
#               help="Num MC files (MC only)")
#p.add_argument("--neventsperfile", type=float, default=None,
#               help="Num events generated per file (MC only)")

args = p.parse_args()

if args.output[-4:] != ".npy":
    raise ValueError("Output file path must end with .npy")
if args.hdf:
    print('Warning: HDF to Numpy array has not been tested and may not work as expected.')

########
# ARGS #
########

###############################################################################

###########
# CORRECT #
###########

data = proc(args.input, args.time, args.fix_leap, args.MC, args.hdf)#, args.nfiles, args.neventsperfile)

data.sort(order='time')
print("Total events found in final array: {}".format(len(data)))

print("\nSaving to {}".format(args.output))
np.save(args.output, data)

###########
# CORRECT #
###########

