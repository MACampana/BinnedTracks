import argparse
from I3Tray import I3Tray
from icecube import hdfwriter, simclasses, recclasses, dataclasses

p = argparse.ArgumentParser(description="Performs i3 -> hdf for simweight-ing",
                            formatter_class=argparse.RawTextHelpFormatter)
p.add_argument("--input", type=str, nargs='+', required=True, help="Input I3 file(s)")

args = p.parse_args()

files = sorted(args.input)

def custom_filter(frame):
    if frame.Stop != icetray.I3Frame.Physics:
        return False
    elif frame['I3EventHeader'].sub_event_stream == 'NullSplit':
        return False
    elif ('MPEFit' not in frame.keys()) and ('SplineMPE' not in frame.keys()):
        return False
    elif not frame['FilterMask']['MuonFilter_13'].condition_passed:
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
          'SplineMPE', 'SplineMPEMuEXDifferential', 'MPEFitParabaloid'],
    output="Level2_IC86.2016_NuMu.021217.hdf5",
)
tray.AddModule('TrashCan', 'YesWeCan')
tray.Execute()