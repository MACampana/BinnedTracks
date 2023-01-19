
import os
import argparse
import tables
import pickle
import numpy as np
import numpy.lib.recfunctions as rf
import scipy.interpolate as interp

from icecube import astro, paraboloid
from icecube import dataio, icetray
from icecube.ps_processing import __version__ as ps_version

exp_dtype = [('run', int), ('event', int), ('subevent', int),
             ('ra', float), ('dec', float),
             ('azi', float), ('zen', float), ('time', float),
             ('logE', float), ('angErr', float),
             ('angErr_noCorrection', float), ('passed_icetopVeto', bool]

mc_dtype = [('trueRa', float), ('trueDec', float),
            ('trueAzi', float), ('trueZen', float),
            ('trueE', float), ('ow', float)]

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

def ow(pint, ptype, generator, trueE, trueZen):
    r""" Compute OneWeight/nevents

    Parameters
    ----------
    pint : float, array
        Interaction probability
    ptype : int, array
        Particle type
    generator : GenerationProbabilityCollection
        Generator probability object from simulation weighting project
    trueE : float, array
        True energy of the primary in GeV
    trueZen : float, array
        True zenith of the primary in radians
    
    Returns
    -------
    ow : float, array
        Simulation OneWeight/nevents
    """
    unit = icetray.I3Units.cm2 / icetray.I3Units.m2
    # NOTE: 2.0 is to account for neutrino + anti-neutrino
    return (pint / unit) / generator(trueE, ptype, np.cos(trueZen)) / 2.0

def hdf2npy(path, generator, assign_time, fix_leap):
    r""" Read events from hdf5 file and return numpy array

    Parameters
    ----------
    path : str
        Path to hdf5 file
    assign_time : float
        Time to assign to events (MC only)
    fix_leap : bool
        Fix leap second bug when true (only for pass1 & pass2)

    Returns
    -------
    a : np.ndarray
        Events formatted as numpy array
    """

    t = tables.open_file(path)
    n = t.root.I3EventHeader.col('Event').size

    if generator is None:
        dtype = exp_dtype
    else:
        dtype = exp_dtype + mc_dtype
    a = np.empty(n, dtype)

    # return empty array if path is empty
    if a.size == 0:
        t.close()
        return a

    a['run'] = t.root.I3EventHeader.col('Run')
    a['event'] = t.root.I3EventHeader.col('Event')
    a['subevent'] = t.root.I3EventHeader.col('SubEvent')
    a['zen'] = t.root.SplineMPE_l4.col('zenith')
    a['azi'] = t.root.SplineMPE_l4.col('azimuth')
    if assign_time is None:
        a['time'] = t.root.I3EventHeader.col('time_start_mjd')
        if fix_leap:
            a['time'][leap_runs(a['run'])] += leap_dt
    else:
        a['time'] = assign_time
    a['logE'] = np.log10(t.root.SplineMPEMuEXDifferential.col('energy'))
    a['ra'], a['dec'] = astro.dir_to_equa(a['zen'], a['azi'], a['time'])

    sig1 = np.hypot(
        t.root.SplineMPE_l4ParaboloidFitParams.col('err1'), 
        t.root.SplineMPE_l4ParaboloidFitParams.col('err2')) / np.sqrt(2)
    sig2 = t.root.SplineMPEBootstrapVectStats.col('median')

    mask = (t.root.SplineMPE_l4ParaboloidFitParams.col('status') == 0)
    a['angErr'][mask] = sig1[mask] # successful fits use paraboloid
    a['angErr'][~mask] = sig2[~mask] # failed fits use median

    if generator is not None:
        # MC fields
        a['trueZen'] = t.root.I3MCWeightDict.col('PrimaryNeutrinoZenith')
        a['trueAzi'] = t.root.I3MCWeightDict.col('PrimaryNeutrinoAzimuth')
        a['trueE'] = t.root.I3MCWeightDict.col('PrimaryNeutrinoEnergy')

        a['trueRa'], a['trueDec'] = astro.dir_to_equa(
            a['trueZen'], a['trueAzi'], a['time'])

        ptype = t.root.I3MCWeightDict.col('PrimaryNeutrinoType')
        ptype = np.array(map(int, ptype)).astype(np.int32)
        try:
            pint = t.root.I3MCWeightDict.col('TotalWeight')
        except:
            pint = t.root.I3MCWeightDict.col('TotalInteractionProbabilityWeight')
        a['ow'] = ow(pint, ptype, generator, a['trueE'], a['trueZen'])

    t.close()
    return a

# END hdf2npy()

def i3f2npy(path, generator, assign_time, fix_leap):
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
    pint, ptype = [], []
    pass_veto = []

    i3f = dataio.I3File(path)
    counter = 0
    for frame in i3f:
        if frame.Stop != icetray.I3Frame.Physics:
            continue
        counter +=1 

        run.append(frame['I3EventHeader'].run_id)
        event.append(frame['I3EventHeader'].event_id)
        subevent.append(frame['I3EventHeader'].sub_event_id)

        zen.append(frame['SplineMPE_l4'].dir.zenith)
        azi.append(frame['SplineMPE_l4'].dir.azimuth)
        if assign_time is None:
            time.append(frame['I3EventHeader'].start_time.mod_julian_day_double)
            if fix_leap and leap_runs(run[-1]):
                time[-1] += leap_dt
        else:
            time.append(assign_time)
        logE.append(np.log10(frame['SplineMPEMuEXDifferential'].energy))

        if frame['SplineMPE_l4Paraboloid'].fit_status == 0:
            angErr.append(np.hypot(
                frame['SplineMPE_l4ParaboloidFitParams'].pbfErr1, 
                frame['SplineMPE_l4ParaboloidFitParams'].pbfErr2) / np.sqrt(2))
        else:
            angErr.append(frame['SplineMPEBootstrapVectStats']['median'])

        if frame.Has("IceTopVeto"):
            pass_veto.append(not frame["IceTopVeto"].value)
        else: 
            pass_veto.append(True)

        if generator is not None:
            # MC fields
            trueZen.append(frame['I3MCWeightDict']['PrimaryNeutrinoZenith'])
            trueAzi.append(frame['I3MCWeightDict']['PrimaryNeutrinoAzimuth'])
            trueE.append(frame['I3MCWeightDict']['PrimaryNeutrinoEnergy'])

            ptype.append(np.int32(frame['I3MCWeightDict']['PrimaryNeutrinoType']))
            try:
                pint.append(frame['I3MCWeightDict']['TotalWeight'])
            except:
                pint.append(frame['I3MCWeightDict']['TotalInteractionProbabilityWeight'])

    # END for (frame)
    print("Found {} p frames".format(counter))

    if generator is None:
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
    a['passed_icetopVeto'] = pass_veto

    if generator is not None:
        # MC fields
        a['trueZen'] = trueZen
        a['trueAzi'] = trueAzi
        a['trueE'] = trueE

        a['trueRa'], a['trueDec'] = astro.dir_to_equa(
            a['trueZen'], a['trueAzi'], a['time'])

        a['ow'] = ow(np.array(pint), np.array(ptype),
                     generator, a['trueE'], a['trueZen'])

    i3f.close()
    return a

# END i3f2npy()

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
    mask = ((data["angErr"] < np.radians(5)) &
            (data["dec"] < np.radians(-5))) + \
           ((data["angErr"] < np.radians(15)) &
            (data["dec"] >= np.radians(-5)))
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

def final_level(f, spline, generator, time, fix_leap):
    r""" Perform pull correction on either hdf5 or i3 file

    Parameters
    ----------
    f : str
        Path to hdf5 or i3 file with events
    spline : str
        Path to pull correction spline
    time : float
        Time to assign to events (MC only)
    fix_leap : bool
        Fix leap second bug when true (only for pass1 & pass2)

    Returns
    -------
    data : np.ndarray
       Array with pull corrected data events
    """
    print("\nPull correcting {}".format(f))
    print("Using {}".format(spline))

    # load pull correction spline
    try:
        spline = np.load(spline, allow_pickle=True, encoding='bytes')
    except:
        spline = np.load(spline)

    # load data file
    data = []
    for filename in f:
        print("Extracting from ", filename)
        if os.path.splitext(filename)[-1] in [".hd5", ".hdf5"]:
            x = hdf2npy(filename, generator, time, fix_leap)
        else:
            x = i3f2npy(filename, generator, time, fix_leap)

        if len(data) == 0: data = x
        else: data = np.concatenate([data, x])

    if data.size:
        # add an angular error field with no correction
        data['angErr_noCorrection'] = data['angErr'].copy()

        # And then apply the correction
        p50E = lambda : interp.splev(data['logE'], spline)
        data['angErr'] *= p50E() / 1.1774
        data = mask_exp(data)

        if generator is not None:
            # apply additional mask to Monte Carlo events
            data = mask_mc(data)

    return data

###############################################################################

########
# ARGS #
########

p = argparse.ArgumentParser(description="Performs final level pull"
                            " correction, cuts, adds MC variables",
                            formatter_class=argparse.RawTextHelpFormatter)
p.add_argument("--upgoing", default=[], action='append', help="Upgoing hdf5 file")
p.add_argument("--downgoing", default=[], action='append', help="Downgoing hdf5 file")
p.add_argument("--output", type=str, required=True, help="Output numpy file")
p.add_argument("--time", type=float, default=None,
               help="Time to assign to events (MC only)")
p.add_argument("--mcfiles", type=str, nargs="+", default=[],
               help="List of simulation file numbers to use given as run:nfile")
p.add_argument("--fix-leap", action='store_true', dest="fix_leap",
               help="Apply leap second bug fix")
p.add_argument("--run", type=int, default=-1, help='Run number')

# GRL-specific arguments
p.add_argument("--output_grl", type=str, help="Output GRL numpy file")
p.add_argument("--good_start_times", type=float, nargs="+", default=[],
               help="List of good start MJDs")
p.add_argument("--good_end_times", type=float, nargs="+", default=[],
               help="List of good end MJDs")

args = p.parse_args()

print("\npython final_level.py \\")
if args.fix_leap:
    print("  --fix-leap \\")
if args.time is not None:
    print("  --time {:.11f} \\".format(args.time))
if len(args.mcfiles):
    args.mcfiles = np.atleast_1d(args.mcfiles)
    print("  --mcfiles", len(args.mcfiles), args.mcfiles)
print("  --upgoing \"{}\" \\".format(args.upgoing))
print("  --downgoing \"{}\" \\".format(args.downgoing))
print("  --output \"{}\"".format(args.output))
print("  --good_start_times \"{}\"".format(args.good_start_times))
print("  --good_end_times \"{}\"".format(args.good_end_times))

if args.output[-4:] != ".npy":
    raise ValueError("Output file path must end with .npy")

if args.upgoing == "" and args.downgoing == "":
    raise ValueError("Must provide at least upgoing or downgoing file")

if not len(args.good_start_times)==len(args.good_end_times):
    raise ValueError("List of GRL start and end times must have the same length")

########
# ARGS #
########

###############################################################################

###########
# CORRECT #
###########

print("\nPS Processing: {}".format(ps_version))

# generator for simulation event probabilities
generator = None
for item in args.mcfiles:

    from icecube.weighting.weighting import from_simprod

    simid = int(item.split(":")[0])
    nfile = int(item.split(":")[1])

    # The file should *really* exist already...
    generator_dir = "/cvmfs/icecube.opensciencegrid.org/users/NeutrinoSources/weighting_generators/"
    generator_file = os.path.join(generator_dir, str(simid)+".pckl")
    current_generator = nfile * pickle.load(open(generator_file, 'rb'))

    if generator is None:
        #generator = nfile * from_simprod(simid)
        generator = current_generator
    else:
        #generator += nfile * from_simprod(simid)
        generator += current_generator

# END for (item)

data = []
path = os.path.expandvars("$I3_BUILD/ps_processing/resources/pull")

if len(args.upgoing):
    # perform pull correction on upgoing events and append to data
    data.append(final_level(args.upgoing,
                            path + "/upgoing_spline_gamma2.0.npy",
                            generator, args.time, args.fix_leap))
if len(args.downgoing):
    # perform pull correction on downgoing events and append to data
    data.append(final_level(args.downgoing,
                            path + "/downgoing_spline_gamma2.0.npy",
                            generator, args.time, args.fix_leap))

# combine and sort by event time
data = np.concatenate(data)
data.sort(order='time')
print("Total events found in final array: {}".format(len(data)))

print("\nSaving to {}".format(args.output))
np.save(args.output, data)

# Create the GRL if requested
if args.output_grl and not len(args.mcfiles):
    grl = []
    for i, (start, end) in enumerate(zip(args.good_start_times, args.good_end_times)):
        events = data[(start <= data['time']) & (data['time'] < end)]
        entry = (float(args.run) + i/1000., start, end, end-start, len(events))
        grl.append(entry)

    grl = np.array(grl, dtype=grl_dtype)

    print("\nSaving GRL entries to {}".format(args.output_grl))
    np.save(args.output_grl, grl)

###########
# CORRECT #
###########

