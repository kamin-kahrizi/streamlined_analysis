from processor import Processor
from raw_data import RawData

import offsetters
import referencers
import smoothers
import rangers
import metrics
import normalizers

import matplotlib.pyplot as plt
import numpy as np

#r = RawData('test_data/020223_S3_Bottom_T0_A_2023_02_02_13_59_48.csv')

r = RawData('~/Google Drive/Shared drives/Nanopath Team/Uri Tayvah/Python Handoff/NpDataAnalysis/Test Data/cameron_test/020223_S3_Bottom_waterinitial_C_2023_02_02_11_06_44.csv')  

bg = r.bg_raw
wl = r.wl

spectra = r.spectra()
spectrum = [x for x in spectra if x['sensor'] == '5'][0]['spectrum']



p0 = Processor({'wavelength': wl,
               'spectra': {
                   'bkg': bg, 'smp': spectrum
                   }
               },
               [
                   offsetters.global_offset,
                   referencers.ln_abs,
                   smoothers.loess_smoother(0.0125),
                   rangers.wavelength(600, 900),
                   normalizers.normalize_to_0_1,
                   metrics.peak_pos
                ]
               )

ax = plt.subplot(2,1,1)

pks = []
centroids = []
offsets =  list(np.arange(0.1,1.1, 0.2))
for offset in offsets:
    p1 = Processor({'wavelength': wl,
                   'spectra': {
                       'bkg': bg, 'smp': spectrum
                       }
                   },
                   [
                       offsetters.add_const(offset),
                       referencers.ln_abs,
                       smoothers.loess_smoother(0.0125),
                       rangers.wavelength(600, 900),
                       normalizers.normalize_to_0_1,
                       metrics.peak_pos
                    ]
                   )
    p2 = Processor({'wavelength': wl,
                   'spectra': {
                       'bkg': bg, 'smp': spectrum
                       }
                   },
                   [
                       offsetters.add_const(offset),
                       referencers.ln_abs,
                       smoothers.loess_smoother(0.0125),
                       rangers.wavelength(600, 900),
                       normalizers.normalize_to_0_1,
                       metrics.centroid
                    ]
                   )

    abs_1 = p1.intermediates[4]['data_dict']
    ax.plot(abs_1['wavelength'], abs_1['spectra']['abs'], label = '{:0.1f}'.format(offset),
             color = plt.cm.cool(offset))
    pks.append(p1.result)
    centroids.append(p2.result)
ax.legend(title = 'Offset')

plt.xlabel('Wavelength (nm)')
plt.ylabel('Abs. (norm.)')

plt.subplot(2,2,3)
plt.plot(offsets, pks, 'k')
for i in range(0, len(offsets)):
    plt.plot(offsets[i], pks[i], '.', color = plt.cm.cool(offsets[i]),
             markersize = 10)
plt.xlabel('Offset (a.u.)')
plt.ylabel('Peak position (nm)')

plt.subplot(2,2,4)
plt.plot(offsets, centroids, 'k')
for i in range(0, len(offsets)):
    plt.plot(offsets[i], centroids[i], '.', color = plt.cm.cool(offsets[i]),
             markersize = 10)

plt.xlabel('Offset (a.u.)')
plt.ylabel('Centroid (nm)')

plt.tight_layout()
plt.show()

#abs_1 = p1.intermediates[4]['data_dict']
#abs_0 = p0.intermediates[4]['data_dict']
#plt.plot(abs_0['wavelength'], abs_0['spectra']['abs'], label = 'v0.0')
#plt.plot(abs_1['wavelength'], abs_1['spectra']['abs'], label = 'v0.1 (+1)')

#plt.legend()
#plt.show()
