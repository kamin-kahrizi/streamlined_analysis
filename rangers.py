import re
import numpy as np

def wavelength(min_wl, max_wl):
    def cut(data_dict):
        spectra_cut = {}
        
        wl = data_dict['wavelength']

        inds = (wl >= min_wl) & ( wl <= max_wl)

        for key, value in data_dict['spectra'].items():
            spectra_cut[key] = value[inds]


        data_dict['spectra'] = spectra_cut
        data_dict['wavelength'] = wl[inds]
        return data_dict
    cut.__name__ = f'wl_range_{min_wl}_{max_wl}'
    return cut


def pct_max(pct):
    def inner(data_dict):
        wl = data_dict['wavelength']

        # cut based on abs fwhm
        abs_spec = data_dict['spectra']['abs']

        inds = abs_spec >= (max(abs_spec)-(np.ptp(abs_spec)*(1-(pct/100)))) 

        spectra_cut = {'abs' : abs_spec[inds]}
        data_dict['wavelength'] = wl[inds]
        data_dict['spectra'] =  spectra_cut
        return data_dict
    inner.__name__ = re.sub(r'\.', '_', f'pct_max_{pct}')
    return inner


