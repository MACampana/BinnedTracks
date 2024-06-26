import argparse
import numpy as np
from icecube.icetray import I3Tray
from icecube import icetray, hdfwriter, simclasses, recclasses, dataclasses, paraboloid

p = argparse.ArgumentParser(description="Performs i3 -> hdf for simweight-ing",
                            formatter_class=argparse.RawTextHelpFormatter)
p.add_argument("--input", type=str, nargs='+', required=True, help="Input I3 file(s)")
p.add_argument("--output", type=str, required=True, 
               help="Output HDF5 file (including extenstion hdf5). Recommended to include number of input files in file name.")

args = p.parse_args()

if args.output[-5:] != ".hdf5":
    raise ValueError("Output file path must end with .hdf5")

files = sorted(args.input)
try:
    print('Removing problem files from input list...')
    files.remove('/data/ana/PointSource/muon_level3/sim/IC86.2016/21220/Level3_IC86.2016_NuMu.021220.001271.i3.zst')
except ValueError:
    print('Problem files not found in input list...continuing...')
    pass

def custom_filter(frame):
    if frame.Stop != icetray.I3Frame.Physics:
        return False
    elif frame['I3EventHeader'].sub_event_stream == 'NullSplit':
        return False
    elif 'SplineMPE' not in frame.keys():
        return False
    elif 'SplineMPEMuEXDifferential' not in frame.keys():
        return False
    elif 'SRTHVInIcePulses_Qtot' not in frame.keys():
        return False
    elif 'SRTHVInIcePulses_QtotWithDC' not in frame.keys():
        return False
    elif 'MPEFitParaboloidFitParams' not in frame.keys():
        return False
    elif np.isnan(frame['SplineMPE'].dir.azimuth) or np.isnan(frame['SplineMPE'].dir.zenith):
        return False
    else:
        return True

tray = I3Tray()
tray.Add("I3Reader", FileNameList=files)
tray.Add(custom_filter)
tray.Add(
    hdfwriter.I3HDFWriter,
    SubEventStreams=["InIceSplit", "Final"],
    keys=["MCPrimary", 'MCPrimary1', "I3MCWeightDict", 'I3EventHeader', 
          'MPEFit', 'MPEFitMuEX', 'MPEFitCramerRaoParams', 
          'SplineMPE', 'SplineMPEMuEXDifferential', 'MPEFitParabaloid', 'MPEFitParaboloidFitParams', 
          'Homogenized_QTot', 'SRTHVInIcePulses_Qtot', 'SRTHVInIcePulses_QtotWithDC'],
    output=args.output,
)
tray.AddModule('TrashCan', 'YesWeCan')
tray.Execute()
