import argparse
import numpy as np
from icecube.icetray import I3Tray, I3Frame

from icecube.level3_filter_muon.level3_Functions import CleanInputStreams
from icecube.level3_filter_muon.level2_5_Cuts import DoPrecuts

from icecube import icetray, hdfwriter, simclasses, recclasses, dataclasses

p = argparse.ArgumentParser(description="Performs i3 -> hdf for simweight-ing",
                            formatter_class=argparse.RawTextHelpFormatter)
p.add_argument("--input", type=str, nargs='+', required=True, help="Input I3 file(s)")
p.add_argument("--output", type=str, required=True, 
               help="Output HDF5 file (including extenstion hdf5). Recommended to include number of input files in file name.")

args = p.parse_args()

if args.output[-5:] != ".hdf5":
    raise ValueError("Output file path must end with .hdf5")

gcd = '/cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.i3.gz'
files = [gcd] + sorted(args.input)

def custom_filter(frame):
    if frame.Stop != icetray.I3Frame.Physics:
        return False
    elif frame['I3EventHeader'].sub_event_stream == 'NullSplit':
        return False
    elif ('MPEFit' not in frame.keys()) and ('SplineMPE' not in frame.keys()):
        return False
    elif not frame['FilterMask']['MuonFilter_13'].condition_passed:
        return False
    elif np.isnan(frame['MPEFit'].dir.azimuth) or np.isnan(frame['MPEFit'].dir.zenith):
        return False
    else:
        return True

tray = I3Tray()
tray.Add("I3Reader", FileNameList=files)
tray.Add(custom_filter)
tray.AddSegment(CleanInputStreams)
tray.AddSegment(DoPrecuts,
        Pulses="SRTInIcePulses",
        Suffix="",
        If=lambda frame: True,
        Track='MPEFit')
tray.Add(
    hdfwriter.I3HDFWriter,
    SubEventStreams=["InIceSplit", "Final"],
    keys=["MCPrimary", 'MCPrimary1', "I3MCWeightDict", 'I3EventHeader', 
          'MPEFit', 'MPEFitMuEX', 'MPEFitCramerRaoParams', 
          'SplineMPE', 'SplineMPEMuEXDifferential', 'MPEFitParabaloid', 'Homogenized_QTot'],
    output=args.output,
)
tray.AddModule('TrashCan', 'YesWeCan')
tray.Execute()
