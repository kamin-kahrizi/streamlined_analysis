import numpy as np
import sys

def ln_abs(data_dict):
    bkg = data_dict['spectra']['bkg']
    smp = data_dict['spectra']['smp']

    data_dict['spectra'] = {'abs': -np.log(smp/bkg)}

    return data_dict

def log10_abs(data_dict):
    bkg = data_dict['spectra']['bkg']
    smp = data_dict['spectra']['smp']
    data_dict['spectra'] = {'abs': -np.log10(smp/bkg)}

    return data_dict

def instrument_spec(data_dict):
    bkg = data_dict['spectra']['bkg']
    smp = data_dict['spectra']['smp']

    spec = smp/bkg
    spec[spec == np.inf] = sys.float_info.max
    spec[spec == -np.inf] = sys.float_info.min

    data_dict['spectra'] = {'abs': spec}
    
    return data_dict
