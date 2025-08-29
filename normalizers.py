import numpy as np

def normalize_to_0_1(data_dict):
    spectra_normed = {}

    for key, value in data_dict['spectra'].items():
        min_val = np.amin(value)
        max_val = np.amax(value-min_val)
        spectra_normed[key] = (value-min_val)/max_val

    data_dict['spectra'] = spectra_normed
    return data_dict

def normalize_to_peak(data_dict):
    spectra_normed = {}

    for key, value in data_dict['spectra'].items():
        max_val = np.amax(value)
        spectra_normed[key] = value/max_val

    data_dict['spectra'] = spectra_normed
    return data_dict



