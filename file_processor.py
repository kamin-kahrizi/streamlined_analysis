from .processor import Processor
from .raw_data import RawData

from pathlib import Path

def process_file(fname, fns_process, fns_record = []):
    # fns_process are used in processor, fns_record is a tuple of (name, function)
    # the function takes the fname and the result of processing, and the value it returns
    # is stored with key <name> in the dictionary returned
    # e.g. ('ri', lambda x: int(re.match(r".*_RI([0-9]*)[R_].*", x).group(1))/100) 
    # would extract the refractive index value from a filename and add it to results
    # with key 'ri'

    r = RawData(fname)
    wl = r.wl
    bkg = r.bg_raw
    
    results = []
    for spec in r.spectra():
        smp = spec['spectrum']
        sensor = spec['sensor']
        
        p = Processor({'wavelength': wl, 'spectra': {'bkg': bkg, 'smp': smp}}, 
                      fns_process)

        result = {
                  'fname': Path(fname).stem,
                  'sensor': sensor,
                  'result': p.result,
                  'method': p.name
                  }
        for name, fn in fns_record:
            result = result | {name: fn(fname, p)}

        results.append(result)

    return results
