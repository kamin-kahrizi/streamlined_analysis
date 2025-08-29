import re
import statsmodels.api as sm
import numpy as np
import time
from scipy.optimize import curve_fit


def weight(vec):
    def smooth(data_dict):
        wl = data_dict['wavelength']
        spectra_smoothed = {}

        for key, value in data_dict['spectra'].items():
            spectra_smoothed[key] = vec*value

        data_dict['spectra'] = spectra_smoothed
        return data_dict
    smooth.__name__ = f'vec_weighted_{int(time.time())}'
    return smooth

def loess_smoother(frac, nm = None): 
    def smooth(data_dict):
        wl = data_dict['wavelength']
        spectra_smoothed = {}

        if nm is not None:
            smooth_frac = nm/np.ptp(wl)
            if smooth_frac > 1:
                smooth_frac = 0
        else:
            smooth_frac = frac

        for key, value in data_dict['spectra'].items():
            spectra_smoothed[key] = sm.nonparametric.lowess(
                    value, wl, smooth_frac)[:,1]

        data_dict['spectra'] = spectra_smoothed
        return data_dict

    if nm is not None:
        smooth.__name__ = re.sub(r'\.', '_', f'loess_{nm}nm')
    else:
        smooth.__name__ = re.sub(r'\.', '_', f'loess_{frac*100:0.2f}')
    return smooth


def polynomial(deg = 2, wl_upsampled = None):
    def smooth(data_dict):
        nonlocal wl_upsampled

        wl = data_dict['wavelength'].astype('float')
        spectra_smoothed = {}

        for key, value in data_dict['spectra'].items():
            p_fit = np.polynomial.Polynomial.fit(wl, value, deg)
            if wl_upsampled is not None:
                wl = wl_upsampled
            spectra_smoothed[key] = np.array([p_fit(x) for x in wl])

        data_dict['spectra'] = spectra_smoothed
        data_dict['wavelength'] = wl
        return data_dict

    smooth.__name__ = re.sub(r'\.', '_', f'poly_{deg:02}')
    return smooth


def triple_gaussian(x, a1, pos_1, sig_1, a2, dpos_2, sig_2, a3, dpos_3, sig_3, y0):
    gauss_1 = a1*np.exp(-(x-pos_1)**2/(2*sig_1**2))
    gauss_2 = a2*np.exp(-(x-(pos_1+dpos_2))**2/(2*sig_2**2))
    gauss_3 = a3*np.exp(-(x-(pos_1+dpos_2 + dpos_3))**2/(2*sig_3**2))
    return gauss_1 + gauss_2 + gauss_3 + y0


def tri_gauss():
    def smooth(data_dict):
        wl = data_dict['wavelength'].astype('float')

        spectra_smoothed = {}
        for key, value in data_dict['spectra'].items():
            p0 = [0.03, 620, 40,
                  0.1,  800-620, 100,
                  0.01, 950-800, 100, 0.04]

            p_opt, _= curve_fit(triple_gaussian, wl, value,
                                p0 = p0, ftol = 1e-6, maxfev=20000)

            spectra_smoothed[key] = triple_gaussian(wl, *p_opt)

        data_dict['spectra'] = spectra_smoothed
        return data_dict

    smooth.__name__ = "tri_gauss"
    return smooth


