import numpy as np
import copy
import matplotlib.pyplot as plt

def gaussian_basis(x, sig, n, offset= None):
    offset = offset or 0
    bpos = np.linspace(min(x)+offset, max(x)-offset, n)
    x = np.reshape(x, (len(x),1))
    
    x = np.tile(x,(1,n)) - bpos
    b = np.exp(-x**2/(2*sig**2))
    b = b/sum(b[:,1])
    return b, bpos

def gaussian_leds(n_leds, fwhm):
    def conv(data_dict):
        wl = data_dict['wavelength']
        sig = fwhm/2.355

        basis, pos = gaussian_basis(wl, sig, n_leds, offset = 75)
        spec = data_dict['spectra']['abs']
        #plt.plot(wl, spec, label = 'Raw', color = (0.7,0.7,0.7))
        led_vals = spec@basis

        data_dict['wavelength'] = pos
        data_dict['spectra']['abs'] = led_vals
        
        #plt.xlabel('Wavelength (nm)')
        #plt.ylabel('Absorbance (a.u.)')
        #plt.plot(pos, led_vals, 'ro', label = 'LEDs')
        #plt.gcf().set_dpi(300)
        #plt.gcf().set_size_inches(6,4)

        return data_dict
    conv.__name__ = f'gauss_leds_x{n_leds}_fwhm_{fwhm}'
    return conv



def poly(order):
    def smooth(data_dict):
        wl = np.arange(700, 1000, 0.2)

        wl_fit = data_dict['wavelength']
        leds = data_dict['spectra']['abs']

        p_fit = np.polynomial.Polynomial.fit(wl_fit, leds, order)
        data_dict['spectra']['abs'] = np.array([p_fit(x) for x in wl])
        data_dict['wavelength'] = wl

        return data_dict

    smooth.__name__ = f'poly_{order}'
    return smooth
