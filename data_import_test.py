from NpDatFuncs_v0_2 import*
from pathlib import Path
from raw_data import RawData

test_data_path = Path('test_data') 

for f in test_data_path.glob('*.csv'):
    r = RawData(f)
    print(r.spectra())


for f in test_data_path.glob('*.csv'):
    Min_wl = 600
    Max_wl = 900
    Smooth = 80
    Norm_Abs = True
    Analyze_file(f,Min_wl,Max_wl,Smooth,Norm_Abs)
