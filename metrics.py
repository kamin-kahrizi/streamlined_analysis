import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def peak_pos(data_dict):
    assert len(data_dict['spectra'].items()) == 1

    spectrum = list(data_dict['spectra'].values())[0]
    wl = data_dict['wavelength']

    peak_ind = (np.diff(np.sign(np.diff(spectrum))) < 0).nonzero()[0] + 1
    if len(peak_ind) == 0:
        return np.nan

    else:
        return wl[peak_ind][np.where(spectrum[peak_ind] == np.amax(spectrum[peak_ind]))][0]

def centroid(data_dict):
    assert len(data_dict['spectra'].items()) == 1

    spectrum = list(data_dict['spectra'].values())[0]
    wl = data_dict['wavelength']

    spectrum_bl = spectrum - np.amin(spectrum)
    return np.sum(spectrum_bl*wl)/np.sum(spectrum_bl)


def centroid_no_bl(data_dict):
    assert len(data_dict['spectra'].items()) == 1
    
    spectrum = list(data_dict['spectra'].values())[0]
    wl = data_dict['wavelength']

    return np.sum(spectrum*wl)/np.sum(spectrum)

def curvature(data_dict):
    wl = data_dict['wavelength']
    spec_abs = data_dict['spectra']['abs']
    p = np.polynomial.Polynomial.fit(wl, spec_abs, 2)
    return p.deriv(2)(0)


def fwhm(data_dict):
    wl = data_dict['wavelength']
    spec_abs = data_dict['spectra']['abs']

    hm = np.max(spec_abs) - (np.ptp(spec_abs)/2)
    overall_peak = wl[spec_abs == np.max(spec_abs)]

    wl_hi = overall_peak
    wl_lo = overall_peak

    for wl_curr in sorted(wl[wl >= overall_peak]):
        if spec_abs[wl == wl_curr] >= hm:
            wl_hi = wl_curr
        else:
            break

    for wl_curr in sorted(wl[wl < overall_peak], reverse = True):
        if spec_abs[wl == wl_curr] >= hm:
            wl_lo = wl_curr
        else:
            break

    
    return float(wl_hi - wl_lo)

def average(data_dict):
    wl = data_dict['wavelength']
    spec_abs = data_dict['spectra']['abs']
    return np.mean(spec_abs)

# Average is to centroid as variance is to this
def spec_var(data_dict):
    wl = data_dict['wavelength']
    spec_abs = data_dict['spectra']['abs']

    spectrum_bl = spec_abs - np.amin(spec_abs)
    cent = centroid(data_dict)
    return np.dot(spectrum_bl, (wl - cent)**2)/np.sum(spectrum_bl) 

def spec_std(data_dict):
    return np.sqrt(spec_var(data_dict))

def gaussian_fwhm(data_dict):
    wl = data_dict['wavelength']
    spec_abs = data_dict['spectra']['abs']

    f = lambda x, pk, a, sig, y0, b: a*np.exp(-(x-pk)**2/(2*sig**2)) + y0 + b*x
    
    p_opt, _ = curve_fit(f, wl, spec_abs, [800, 1, 100, 0, 0])
    plt.plot(wl, spec_abs)
    plt.plot(wl, f(wl, *p_opt))
    return np.abs(p_opt[2])
