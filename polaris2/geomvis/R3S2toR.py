import numpy as np
from polaris2.geomvis import util

# List of single dipoles in [[x,y,z,sx,sy,sz]] form. 
class xyzj_list:
    def __init__(self, data, shape=[10,10,2.5], vmin=0, vmax=None, xlabel='',
                 title=''):
        self.data = data
        self.shape = shape # um

        self.vmin = vmin
        self.vmax = vmax
        self.xlabel = xlabel
        self.title = title

    def plot(self, f, fc):
        ax0, ax1, axs3 = util.plot_template(f, fc, shape=self.shape,
                                            xlabel=self.xlabel,
                                            title=self.title)

        ax1.axis('off')
        for ax in axs3:
            ax.plot(0,0, 'xk', markersize=2, color='k') # Plot origin markers
            ax.axis('off')

        # Set data limits
        xs, ys, zs = self.shape
        axs3[0].set_xlim([-xs/2,xs/2])
        axs3[0].set_ylim([-ys/2,ys/2])
        axs3[1].set_xlim([-xs/2,xs/2])
        axs3[1].set_ylim([-zs/2,zs/2])
        axs3[2].set_xlim([-zs/2,zs/2])
        axs3[2].set_ylim([-ys/2,ys/2])
            
        # Plot dipoles
        for dipole in self.data:
            r = 0.1*np.max(self.shape)
            x, y, z, sx, sy, sz = dipole
            axs3[0].annotate('', xy=(x-r*sx,y-r*sy), xytext=(x+r*sx,y+r*sy), arrowprops=dict(arrowstyle="<|-|>", shrinkA=0, shrinkB=0, lw=0.5, facecolor='k'))
            axs3[1].annotate('', xy=(x-r*sx,z-r*sz), xytext=(x+r*sx,z+r*sz), arrowprops=dict(arrowstyle="<|-|>", shrinkA=0, shrinkB=0, lw=0.5, facecolor='k'))
            axs3[2].annotate('', xy=(z-r*sz,y-r*sy), xytext=(z+r*sz,y+r*sy), arrowprops=dict(arrowstyle="<|-|>", shrinkA=0, shrinkB=0, lw=0.5, facecolor='k'))
            
            axs3[0].plot(x, y, 'xk', markersize=2)
            axs3[1].plot(x, z, 'xk', markersize=2)
            axs3[2].plot(z, y, 'xk', markersize=2) 
