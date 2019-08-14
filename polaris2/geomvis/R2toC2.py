import matplotlib as mpl
import numpy as np
from polaris2.geomvis import util
from matplotlib.transforms import Bbox

# Model R2toC2 in the most general case
class xy:
    def __init__(self, data, circle=False, title='', xlabel='', toplabel='',
                 bottomlabel='', colormax=None, fov=1, plotfov=1):
        # [idimx, indimy, {outxcomplex, outycomplex}]
        # For example: 1000x1000x2 array with complex entries
        self.data = data

        self.circle = circle
        self.title = title
        self.xlabel = xlabel
        self.toplabel = toplabel
        self.bottomlabel = bottomlabel

        self.colormax = colormax

        self.fov = fov
        self.plotfov = plotfov

    def plot(self, f, fc):

        # Use for placing the title
        axs = util.plot_template(f, fc, title=self.title, scale_bar=False)
        axs[0].axis('off')
        axs[1].axis('off')
        
        # Custom placement of axes
        fx, fy, fw, fh = fc

        # Set precise positions of axes
        w = 0.375*fw
        h = 0.375*fh

        # Center coordinates
        cx = fx + 0.425*fw
        cy = fy + 0.5*fh

        # Make three axes
        axb = f.add_axes(Bbox([[cx-w,cy-h],[cx,cy]]))
        axt = f.add_axes(Bbox([[cx-w,cy],[cx,cy+h]]))
        axc = f.add_axes(Bbox([[cx+1*w/4,cy-h/2],[cx+5*w/4,cy+h/2]]))
        for ax in [axb, axt, axc]:
            ax.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

        # Scale bar and labels
        scale_shift = 0.05*fh
        axs[0].annotate('', xy=(cx-w,cy-h-scale_shift), xytext=(cx, cy-h-scale_shift), xycoords='figure fraction', textcoords='figure fraction', va='center', arrowprops=dict(arrowstyle="|-|, widthA=0.5, widthB=0.5", shrinkA=0, shrinkB = 0, lw=.75))
        axs[0].annotate(self.xlabel, xy=(1,1), xytext=(cx-w/2,cy-h-0.1*fh), textcoords='figure fraction', ha='center', va='center', rotation=0)
        axs[0].annotate(self.toplabel, xy=(1,1), xytext=(cx+w/8,cy+h/2), textcoords='figure fraction', ha='center', va='center', rotation=0)
        axs[0].annotate(self.bottomlabel, xy=(1,1), xytext=(cx+w/8,cy-h/2), textcoords='figure fraction', ha='center', va='center', rotation=0)
        
        # For labelling color scale
        if self.colormax is None:
            self.colormax = np.max(np.abs(self.data))

        # Color scale axis
        x = np.linspace(-1, 1, 100)
        y = np.linspace(-1, 1, 100)
        xx, yy = np.meshgrid(x, y)
        im = axc.imshow(util.c2rgb(xx + 1j*yy), interpolation='bicubic', extent=[-1,1,-1,1], origin='lower')
        axc.axis('off')
        patch = mpl.patches.Circle((0,0), radius=1, linewidth=0.5, facecolor='none',
                                   edgecolor='k', transform=axc.transData, clip_on=False)
        im.set_clip_path(patch)
        axc.add_patch(patch)

        axc.plot([-1,1],[0,0],':k', lw=0.5)
        axc.plot([0,0],[-1,1],':k', lw=0.5)        
        axc.annotate('Im', xy=(1,1), xytext=(0.5,1.1), textcoords='axes fraction', ha='center', va='center', rotation=0)
        axc.annotate('Re', xy=(1,1), xytext=(1.1,0.5), textcoords='axes fraction', ha='center', va='center', rotation=0)

        axc.annotate('', xy=(0.5,-0.15), xytext=(1, -0.15), xycoords='axes fraction', textcoords='axes fraction', va='center', arrowprops=dict(arrowstyle="|-|, widthA=0.5, widthB=0.5", shrinkA=0, shrinkB = 0, lw=.75))
        axc.annotate('{:.2g}'.format(self.colormax), xy=(0,0), xytext=(0.75, -0.25), textcoords='axes fraction', ha='center', va='center', rotation=0)

        # Plot
        for i, ax in enumerate([axt, axb]):
            ax.set_xlim(self.plotfov)
            ax.set_ylim(self.plotfov)
            im = ax.imshow(util.c2rgb(self.data[:,:,i], rmax=self.colormax), interpolation='nearest',
                           extent=2*self.fov, origin='lower')
            if self.circle:
                ax.axis('off')
                patch = mpl.patches.Circle((0,0), radius=self.plotfov[0], linewidth=0.5, facecolor='none',
                                           edgecolor='k', transform=ax.transData, clip_on=False)
                im.set_clip_path(patch)
                ax.add_patch(patch)
