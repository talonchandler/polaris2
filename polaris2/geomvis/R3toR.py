import numpy as np
from polaris2.geomvis import utilmpl

class xyz:
    def __init__(self, data, vmin=0, vmax=None, xlabel='', title=''):
        self.data = data

        self.vmin = vmin
        self.vmax = vmax
        self.xlabel = xlabel
        self.title = title        

    def plot(self, f, fc):
        ax0, ax1, axs3 = utilmpl.plot_template(f, fc, shape=self.data.shape,
                                            xlabel=self.xlabel,
                                            title=self.title)
