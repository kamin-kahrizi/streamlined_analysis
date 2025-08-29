import numpy as np
import pandas as pd
import re
from collections import namedtuple
import json

Spectrum = namedtuple('Spectrum', ['fname', 'location', 'wavelength', 'spectrum'])

class RawData:
    def __init__(self, fname):
        self.fname = fname

        self.chunks = self.get_chunk_indices()
        # read spectra
        self.df = self.read_spectra()
        self.wl=np.array(self.df['Raw Data / Wavelength'].tolist())
        self.bg_raw=np.array(self.df['Light Source'].tolist())

        # read results of instrument processing
        self.instrument_results = self.read_instrument_results()

        # read algorithm settings for instrument processing 
        self.algo_settings = self.read_algo_settings()

    def get_chunk_indices(self):
        with open(self.fname, 'r') as f:
            pat = re.compile(r"^[\s,]*$")
            
            inds_start = []
            inds_end = []
            blank = True
            for ind, x in enumerate(f.readlines()):
                if pat.match(x):
                    if not blank:
                        inds_end.append(ind)
                    blank = True

                else:
                    if blank:
                        inds_start.append(ind)
                    blank = False
            return list(zip(inds_start, inds_end))

    def read_metadata(self):
        chunk_start, chunk_end = self.chunks[0]
        df=pd.read_csv(self.fname,skiprows = chunk_start,
                       nrows = chunk_end - chunk_start,
                       header = None)
        return {df.loc[x,0]:df.loc[x,1] for x in df.index}

    def read_spectra(self):
        chunk_start, chunk_end = self.chunks[2]
        df=pd.read_csv(self.fname,skiprows = chunk_start,
                       nrows = chunk_end - chunk_start,
                       header = None)
        df=df.T
        df.columns = df.iloc[0]
        df=df.drop(0)
        return df

    def read_instrument_results(self):
        chunk_start, chunk_end = self.chunks[1]
        df = pd.read_csv(self.fname, skiprows = chunk_start+1,
                         nrows = chunk_end - chunk_start-2)
        return df

    def read_algo_settings(self):
        chunk_start, chunk_end = self.chunks[6]
        df = pd.read_csv(self.fname, skiprows = chunk_start+1,
                         nrows = chunk_end - chunk_start-1, header = None)
        df=df.T
        df.columns = df.iloc[0]
        df=df.drop(0)

        return df

    def read_locations(self):
        chunk_start, chunk_end = self.chunks[4]
        df = pd.read_csv(self.fname, skiprows = chunk_start+1,
                         nrows = chunk_end - chunk_start - 2)
        df = df.set_index('#')
        return df

    def spectra_by_location(self):
        locs = self.read_locations() 
        spectra = []
        for smp in locs.index:
            smp_loc = locs['Sample (mm)'].loc[smp]
            bkg_loc = locs['Background (mm)'].loc[smp]

            if smp_loc != 'X':
                spectra.append(Spectrum(fname = self.fname,
                                        location = smp_loc,
                                        spectrum = self.df[f'{smp} Sample'].to_numpy(),
                                        wavelength = self.wl))
            if bkg_loc != 'X':
                for i in [1,2]:
                    col_name = f'{smp} BG{i}'
                    spectra.append(Spectrum(fname = self.fname,
                                            location = bkg_loc,
                                            spectrum = self.df[col_name].to_numpy(),
                                            wavelength = self.wl))

        return spectra


    def spectra(self):
        bkg = self.bg_raw
        spectra = []
        for col in [x for x in self.df.columns if x[0].isdigit()]:
            if 'BG' in col:
                raise NotImplementedError('This appears to be an old file type which is no longer supported. These files can be analyzed with an older version of NpDataAnalysis (< v0.7)')

            match col.split():
                case smp_num, 'Sample':
                    smp = self.df[col].astype(float).to_numpy()
                case _, 'Background':
                    bkg = self.df[col].astype(float).to_numpy()
                    continue
            spectra.append({'spectrum': smp, 
                            'background': bkg,
                            'sensor': smp_num})
        return spectra

    def read_parameters(self):
        with open(self.fname) as f:
            for line in f:
                pass

        return json.loads(
                line.replace('Settings Import Begin:', '')\
                        .replace('Settings Import End', ''))
    
    