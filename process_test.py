import pandas as pd

from processor import Processor

import offsetters
import normalizers
import referencers
import smoothers
import rangers
import metrics

import matplotlib.pyplot as plt

from NpDatFuncs_v0_2 import *

def example_spectrum():
    df = pd.read_csv('test_data/example_spectrum.csv', comment = "#")
    return {'bkg': df['bkg'].to_numpy(), 'smp': df['smp'].to_numpy(),
            'wl': df['wl'].to_numpy()}

def npda_analysis(wl_range, smooth):
    spectrum = example_spectrum()
    Bg,Sg,bg_offset, sg_offset,\
                Abs,Abs_sm,T0,peak_WL,\
                peak_H,Q,steepness,sigma_noise,\
                centrd,Bg_higher, \
                Abs_no_trans, Abs_no_norm, Abs_norm  = \
                DataAnalysisM1(spectrum['bkg'], spectrum['smp'], spectrum['wl'],
                               wl_range, smooth, False)

    return locals()


# mimic NpDataAnalysis with processing pipeline
# check that peak position matches
def test_npda_no_norm_peak():
    spectrum = example_spectrum()

    wl_range = [800, 1000]
    smooth = 1.25

    npda_results = npda_analysis(wl_range, smooth)
    p = Processor({'wavelength': spectrum['wl'],
               'spectra': { 'bkg': spectrum['bkg'], 'smp': spectrum['smp']}},
                  [
                   offsetters.global_offset,
                   normalizers.normalize_to_peak,
                   referencers.log10_abs,
                   smoothers.loess_smoother(smooth/100),
                   rangers.wavelength(wl_range[0], wl_range[1]),
                   metrics.peak_pos
                  ])
   
    # check that unsmoothed matched
    assert npda_results['peak_WL'] == p.result

def test_npda_no_norm_centroid():
    spectrum = example_spectrum()

    wl_range = [800, 1000]
    smooth = 1.25

    npda_results = npda_analysis(wl_range, smooth)
    p = Processor({'wavelength': spectrum['wl'],
               'spectra': { 'bkg': spectrum['bkg'], 'smp': spectrum['smp']}},
                  [
                   offsetters.global_offset,
                   normalizers.normalize_to_peak,
                   referencers.log10_abs,
                   smoothers.loess_smoother(smooth/100),
                   rangers.wavelength(wl_range[0], wl_range[1]),
                   metrics.centroid
                  ])
   
    # check that unsmoothed matched
    assert npda_results['centrd'] == p.result


# mimic NpDataAnalysis with processing pipeline
# check that unsmoothed, transmission normalized matches
def test_npda_no_norm_no_smooth():
    spectrum = example_spectrum()

    wl_range = [800, 1000]
    smooth = 1.25

    npda_results = npda_analysis(wl_range, smooth)
    p = Processor({'wavelength': spectrum['wl'],
               'spectra': { 'bkg': spectrum['bkg'], 'smp': spectrum['smp']}},
                  [
                   offsetters.global_offset,
                   normalizers.normalize_to_peak,
                   referencers.log10_abs
                  ])
   
    # check that unsmoothed matched
    unsmoothed = p.intermediates[2]['data_dict']['spectra']['abs']
    assert np.allclose(unsmoothed, npda_results['Abs_no_norm'])

def test_npda_no_norm_no_smooth_no_trans():
    spectrum = example_spectrum()

    wl_range = [800, 1000]
    smooth = 1.25

    npda_results = npda_analysis(wl_range, smooth)
    p = Processor({'wavelength': spectrum['wl'],
               'spectra': { 'bkg': spectrum['bkg'], 'smp': spectrum['smp']}},
                  [
                   offsetters.global_offset,
                   referencers.log10_abs
                  ])
   
    # check that unsmoothed matched
    unsmoothed = p.intermediates[1]['data_dict']['spectra']['abs']
    assert np.allclose(unsmoothed, npda_results['Abs_no_trans'])

def test_npda_no_norm_smooth():
    spectrum = example_spectrum()

    wl_range = [800, 1000]
    smooth = 1.25

    npda_results = npda_analysis(wl_range, smooth)
    p = Processor({'wavelength': spectrum['wl'],
               'spectra': { 'bkg': spectrum['bkg'], 'smp': spectrum['smp']}},
                  [
                   offsetters.global_offset,
                   normalizers.normalize_to_peak,
                   referencers.log10_abs,
                   smoothers.loess_smoother(smooth/100)
                  ])
   
    # check that smoothed matched
    smoothed = p.intermediates[3]['data_dict']['spectra']['abs']
    assert np.allclose(smoothed, npda_results['Abs_sm'])


