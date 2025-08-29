from processor import Processor
from raw_data import RawData

import offsetters
import referencers
import smoothers
import rangers
import metrics

r = RawData('test_data/020223_S3_Bottom_T0_A_2023_02_02_13_59_48.csv')

bg = r.bg_raw
wl = r.wl

spectra = r.spectra()
spectrum = [x for x in spectra if x['sensor'] == '6'][0]['spectrum']

p = Processor({'wavelength': wl,
               'spectra': {
                   'bkg': bg, 'smp': spectrum
                   }
               },
               [
                   offsetters.global_offset,
                   referencers.ln_abs,
                   smoothers.loess_smoother(0.0125),
                   rangers.wavelength(600, 900),
                   metrics.peak_pos
                ]
               )

print(p.result) 
