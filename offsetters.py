import numpy as np
import functools

def global_offset(data_dict):
    # offset both spectra by the same amount (overall minimum plus overall resolution)
    spectra = data_dict['spectra'].values()
    combined_spectra = functools.reduce(np.union1d, spectra)

    minval = np.amin(combined_spectra)
    res = np.amin(np.amin(np.abs(combined_spectra[np.nonzero(combined_spectra)])))

    spectra_offset = {}
    for key, value in data_dict['spectra'].items():
        spectra_offset[key] = value - minval + res

    data_dict['spectra'] = spectra_offset
    return data_dict

def min_to_zero(data_dict):
    spectra_offset = {}
    for key, value in data_dict['spectra'].items():
        spectra_offset[key] = value - min(value)

    data_dict['spectra'] = spectra_offset
    return data_dict

def add_const(c):
    def offset(data_dict):
       spectra_offset = {}
       for key, value in data_dict['spectra'].items():
           spectra_offset[key] = value + c
       data_dict['spectra'] = spectra_offset
       return data_dict

    return offset
