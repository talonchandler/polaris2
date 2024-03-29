import tifffile
import numpy as np
from polaris2.geomvis import utilmpl
import logging
log = logging.getLogger('log')

class xy:
    def __init__(self, data, px_dims=[1,1], cmap='gray', title='',
                 fov=[0,1], plotfov=[0,1], vmin=None, vmax=None):
                 
        self.data = data
        self.px_dims = px_dims

        self.cmap = cmap
        self.xlabel = '{:.0f}'.format(plotfov[1] - plotfov[0]) + ' $\mu$m'
        self.title = title

        self.fov = fov
        self.plotfov = plotfov

        self.vmin = vmin
        self.vmax = vmax

    def to_tiff(self, filename):
        utilmpl.mkdir(filename)
        with tifffile.TiffWriter(filename, imagej=True) as tif:
            tif.save(self.data.astype(np.float32),
                     resolution=(1/self.px_dims[0], 1/self.px_dims[1]),
                     metadata={'unit':'um'}) # TZCYXS

    def plot(self, f, fc, ss):
        ax = utilmpl.plot_template(f, fc, shape=self.data.shape, xlabel=self.xlabel,
                                   title=self.title)

        # Image
        if self.cmap is 'gray':
            vmax = np.max(self.data)
            vmin = 0
        elif self.cmap is 'bwr':
            vmax = np.max(np.abs(self.data))
            vmin = -vmax

        if self.vmax is not None:
            vmax = self.vmax
            vmin = self.vmin

        ax[0].set_xlim(self.plotfov)
        ax[0].set_ylim(self.plotfov)
        ax[0].imshow(self.data.T, vmin=vmin, vmax=vmax, cmap=self.cmap,
                     extent=2*self.fov,
                     aspect='auto', interpolation='nearest', origin='lower')

        # Colorbar
        x = np.linspace(vmin, vmax, 100)
        xx, yy = np.meshgrid(x, x)

        ax[1].imshow(yy, vmin=vmin, vmax=vmax, cmap=self.cmap,
                     extent=[0,1,vmin,vmax], aspect='auto',
                     interpolation='bicubic', origin='lower')

        if self.cmap is 'gray':
            ax[1].annotate('{:.2g}'.format(np.max(vmax)), xy=(0,0), xytext=(0, 1.05), textcoords='axes fraction', va='center', ha='left')
            ax[1].annotate('0', xy=(0,0), xytext=(1.8, 0), textcoords='axes fraction', va='center', ha='left')
            ax[1].yaxis.set_ticks([0, vmax])
            ax[1].set_yticklabels(['', ''])
        elif self.cmap is 'bwr':
            ax[1].annotate('{:.2g}'.format(vmax), xy=(0,0), xytext=(0, 1.05), textcoords='axes fraction', va='center', ha='left')
            ax[1].annotate('${:.2g}$'.format(vmin), xy=(0,0), xytext=(0, -0.05), textcoords='axes fraction', va='center', ha='left')
            ax[1].yaxis.set_ticks([vmin, 0, vmax])
            ax[1].set_yticklabels(['', '', ''])

        # Colors
        ax[0].annotate('', xy=(0,0), xytext=(0.1, 0), xycoords='axes fraction', textcoords='axes fraction', arrowprops=dict(arrowstyle="-", lw=2, shrinkB=0, color='red'))
        ax[0].annotate('', xy=(0,0), xytext=(0, 0.1), xycoords='axes fraction', textcoords='axes fraction', arrowprops=dict(arrowstyle="-", lw=2, shrinkB=0, color=[0,1,0]))
