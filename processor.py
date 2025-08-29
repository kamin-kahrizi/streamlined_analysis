import matplotlib.pyplot as plt
import inspect
import copy

class Processor:
    def __init__(self, data_dict, functions):
        self.data_dict = data_dict
        self.functions = functions

        self.intermediates = []

        for fn in functions[:-1]:
            data_dict = fn(data_dict)
            self.intermediates.append({
                'data_dict': copy.deepcopy(data_dict),
                'name': fn.__name__,
                'source': inspect.getsource(fn)
                })
        self.result = functions[-1](data_dict)
        name = '_'.join([x.__name__ for x in functions[:-1]])
        name += f'[{functions[-1].__name__}]'
        self.name = name


    def resample(self, raw, smoothed, spectra):
        print()

    def apply_partial(self, fns, name = None, index = None):
        assert (name is not None) or (index is not None)

        if index is None:
            names_to_inds = {}
            for i, intermediate in enumerate(self.intermediates):
                cur_name = intermediate['name']
                assert cur_name not in names_to_inds, f'Ambiguous name ({name})'
                names_to_inds[cur_name] = i

            index = names_to_inds[name]

        data_dict = copy.deepcopy(self.intermediates[index]['data_dict'])
        for fn in fns[:-1]:
            data_dict = fn(data_dict)
        return fns[-1](data_dict)

    def plot(self):
        fig, axs = plt.subplots(1, len(self.intermediates))
        for i in range(0, len(self.intermediates)):
            wl = self.intermediates[i]['data_dict']['wavelength']
            for name, vals in self.intermediates[i]['data_dict']['spectra'].items():
                axs[i].plot(wl, vals, label = name)
            axs[i].legend()
        ax_last = axs[len(self.intermediates) -1]
        ax_last.vlines(self.result, *ax_last.get_ylim())
        ax_last.set_title(self.result)

        return fig





