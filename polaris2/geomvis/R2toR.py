import tifffile
import numpy as np
from polaris2.geomvis import utilmpl
import logging
log = logging.getLogger('log')

class xy:
    def __init__(self, data, vmin=0, vmax=None, xlabel='', title='',
                 fov=1, plotfov=1):
                 
        self.data = data

        self.vmin = vmin
        self.vmax = vmax
        self.xlabel = xlabel
        self.title = title

        self.fov = fov
        self.plotfov = plotfov

    def save_tiff(self, filename='sh.tif'):
        utilmpl.mkdir(filename)
        with tifffile.TiffWriter(filename, imagej=True) as tif:
            tif.save(self.data.astype(np.float32)) # TZCYXS

    def plot(self, f, fc):
        ax = utilmpl.plot_template(f, fc, shape=self.data.shape, xlabel=self.xlabel,
                                title=self.title)

        # Image
        vmin = self.vmin
        if self.vmax is None:
            vmax = np.max(self.data)
        else:
            vmax = self.vmax

        ax[0].set_xlim(self.plotfov)
        ax[0].set_ylim(self.plotfov)
        ax[0].imshow(self.data, vmin=vmin, vmax=vmax, cmap='gray',
                     extent=2*self.fov,
                     aspect='auto', interpolation='nearest', origin='lower')

        # Colorbar
        x = np.linspace(vmin, vmax, 100)
        xx, yy = np.meshgrid(x, x)

        ax[1].imshow(yy, vmin=vmin, vmax=vmax, cmap='gray',
                     extent=[0,1,vmin,vmax], aspect='auto',
                     interpolation='bicubic', origin='lower')
        ax[1].annotate('{:.2g}'.format(np.max(vmax)), xy=(0,0), xytext=(0, 1.05), textcoords='axes fraction', va='center', ha='left')
        ax[1].annotate('0', xy=(0,0), xytext=(1.8, 0), textcoords='axes fraction', va='center', ha='left')        
        ax[1].yaxis.set_ticks([0, vmax])
        ax[1].set_yticklabels(['', ''])

        # Colors
        ax[0].annotate('', xy=(0,0), xytext=(0.1, 0), xycoords='axes fraction', textcoords='axes fraction', arrowprops=dict(arrowstyle="-", lw=2, shrinkB=0, color='red'))
        ax[0].annotate('', xy=(0,0), xytext=(0, 0.1), xycoords='axes fraction', textcoords='axes fraction', arrowprops=dict(arrowstyle="-", lw=2, shrinkB=0, color=[0,1,0]))
