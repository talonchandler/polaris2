import subprocess
import os
import matplotlib as mpl
import matplotlib.figure as fig
from matplotlib.transforms import Bbox
import numpy as np
import logging
log = logging.getLogger('log')

# Make directory if it doesn't exist
def mkdir(filename):
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        log.info('Making folder '+folder)
        os.makedirs(folder)

# Make figure from grid of geomvis objects
def plot(input_objects, output_file='output.pdf', ss=1):
    mkdir(output_file)
    
    rows = len(input_objects)
    cols = len(input_objects[0])

    f = fig.Figure(figsize=(3*cols,3*rows))
    for row in range(rows):
        for col in range(cols):
            fc = [col/cols, (rows-row-1)/rows, 1/cols, 1/rows]
            input_objects[row][col].plot(f, fc, ss)
            
    f.savefig(output_file, dpi=300*ss)

# Plot template for 2D and 3D matplotlib geometries
def plot_template(f, fc, shape=(1,1), xlabel='', ylabel='', clabel='', title='',
                  scale_bar=True, bump=1.0, invert=False):

    # Subfigure coords
    fx, fy, fw, fh = fc
    
    if invert:
        color='white'
        from matplotlib.patches import Rectangle
        f.patches.extend([Rectangle((fx, fy), fw, fh,
                                    fill=True, color='k', zorder=-1,
                                    transform=f.transFigure, figure=f)])
    else:
        color='black'
    
    

    # Set precise positions of axes
    wnom = 0.375*fw
    hnom = 0.375*fh

    # 2D vs 3D
    if len(shape) == 2: # 2D
        xlen = shape[0]
        ylen = shape[1]
    elif len(shape) == 3: # 3D
        xlen = shape[0] + shape[2]
        ylen = shape[1] + shape[2]

    # Width and height
    if ylen == xlen:
        w = wnom
        h = hnom
    elif ylen < xlen:
        w = wnom
        h = hnom*ylen/xlen
    elif ylen > xlen:
        w = wnom*xlen/ylen
        h = hnom

    w = w*bump
    h = h*bump
        
    # Center coordinates
    cx = fx + 0.425*fw*bump
    cy = fy + 0.5*fh

    # Color bar spacing
    cspace = 0.02*fw
    cwidth = 0.04*fw

    # Create axes
    ax0 = f.add_axes(Bbox([[cx-w,cy-h],[cx+w,cy+h]]))
    if len(shape) == 3: # 3D
        xx, yy, zz = shape
        ax0.axis('off')
        axs3 = []
        acx = cx - ((xx - zz)/(xx + zz))*w
        acy = cy - ((yy - zz)/(yy + zz))*h
        sw = 0.015*fw # shift for axis labels
        sh = 0.015*fh
        
        axs3.append(f.add_axes(Bbox([[acx+sw,acy+sh],[cx+w,cy+h]]))) #xy
        axs3.append(f.add_axes(Bbox([[acx+sw,cy-h],[cx+w,acy-sh]]))) #xz
        axs3.append(f.add_axes(Bbox([[cx-w,acy+sh],[acx-sw,cy+h]]))) #yz
        for ax in axs3:
            ax.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

        alenx = 0.1*fw
        aleny = 0.1*fh
        sss = 1.15 # Stretch labels
        ax0.annotate('$x$', xy=(acx,acy), xytext=(acx+sss*alenx,acy), xycoords='figure fraction', textcoords='figure fraction', va='center', ha='center', fontsize=12)
        ax0.annotate('$y$', xy=(acx,acy), xytext=(acx,acy+sss*aleny), xycoords='figure fraction', textcoords='figure fraction', va='center', ha='center', fontsize=12)
        ax0.annotate('$z$', xy=(acx,acy), xytext=(acx-sss*alenx,acy), xycoords='figure fraction', textcoords='figure fraction', va='center', ha='center', fontsize=12)
        ax0.annotate('$z$', xy=(acx,acy), xytext=(acx,acy-sss*aleny), xycoords='figure fraction', textcoords='figure fraction', va='center', ha='center', fontsize=12)
        ax0.annotate('', xy=(acx,acy), xytext=(acx+alenx,acy), xycoords='figure fraction', textcoords='figure fraction', va='center', ha='center', arrowprops=dict(arrowstyle="<-", shrinkB=0, lw=0.5))
        ax0.annotate('', xy=(acx,acy), xytext=(acx,acy+aleny), xycoords='figure fraction', textcoords='figure fraction', va='center', ha='center', arrowprops=dict(arrowstyle="<-", shrinkB=0, lw=0.5))
        ax0.annotate('', xy=(acx,acy), xytext=(acx-alenx,acy), xycoords='figure fraction', textcoords='figure fraction', va='center', ha='center', arrowprops=dict(arrowstyle="<-", shrinkB=0, lw=0.5))
        ax0.annotate('', xy=(acx,acy), xytext=(acx,acy-aleny), xycoords='figure fraction', textcoords='figure fraction', va='center', ha='center', arrowprops=dict(arrowstyle="<-", shrinkB=0, lw=0.5))

    ax1 = f.add_axes(Bbox([[cx+wnom+cspace,cy-hnom],[cx+wnom+cspace+cwidth,cy+hnom]]))

    # Ticks
    ax0.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax1.tick_params(axis='y', which='both', right=True, left=False, labelleft=False, labelright=True)

    # Scale bar on x axis
    scale_shift = 0.05*fh
    if len(shape) == 2:
        if scale_bar:
            ax0.annotate('', xy=(cx-w,cy-h-scale_shift), xytext=(cx+w, cy-h-scale_shift), xycoords='figure fraction', textcoords='figure fraction', va='center', arrowprops=dict(arrowstyle="|-|, widthA=0.5, widthB=0.5", shrinkA=0, shrinkB = 0, lw=.75))
        ax0.annotate(xlabel, xy=(1,1), xytext=(cx, fy + 0.03*fh), textcoords='figure fraction', ha='center', va='center', rotation=0, zorder=10, color=color)
    elif len(shape) == 3:
        if scale_bar:
            ax0.annotate('', xy=(acx+sw,cy-h-scale_shift), xytext=(cx+w, cy-h-scale_shift), xycoords='figure fraction', textcoords='figure fraction', va='center', arrowprops=dict(arrowstyle="|-|, widthA=0.5, widthB=0.5", shrinkA=0, shrinkB = 0, lw=.75))
        ax0.annotate(xlabel, xy=(1,1), xytext=(acx + (cx+w - acx)/2,cy-h-0.1*fh), textcoords='figure fraction', ha='center', va='center', rotation=0)

    # Labels
    ax0.annotate(title, xy=(1,1), xytext=(cx,cy+hnom+0.075*fh), textcoords='figure fraction', ha='center', va='center', color=color)
    ax1.annotate(clabel, xy=(1,1), xytext=(cx+wnom+cspace+cwidth+0.1*fw, cy), textcoords='figure fraction', ha='center', va='center', rotation=270)

    if len(shape) == 2:
        return ax0, ax1
    else:
        return ax0, ax1, axs3

# Convert a folder of pngs into a movie with ffmpeg
# Poorer quality than manual avi conversion from imagej
def video_stitch(in_path, out_name, framerate=10):
    subprocess.call(['ffmpeg', '-nostdin', '-y', '-framerate', str(framerate),
                     '-loglevel', 'panic', '-i', in_path+'%03d.png',
                     '-pix_fmt', 'yuvj420p', '-vcodec', 'rawvideo', out_name])

# Convert complex numbers to rgb
# Based on: https://stackoverflow.com/a/20958684/5854689
def c2rgb(z, rmin=0, rmax=1, hue_start=0):
    # get amplidude of z and limit to [rmin, rmax]
    amp = np.abs(z)
    amp = np.where(amp < rmin, rmin, amp)
    amp = np.where(amp > rmax, rmax, amp)
    ph = np.angle(z, deg=1) + hue_start
    # HSV are values in range [0,1]
    h = (ph % 360) / 360
    s = 0.85 * np.ones_like(h)
    v = (amp -rmin) / (rmax - rmin)
    return mpl.colors.hsv_to_rgb(np.dstack((h,s,v)))

# Labels
def shape2xlabel(shape):
    a = '{:.0f}'.format(shape[0])
    b = '{:.0f}'.format(shape[1])
    c = '{:.0f}'.format(shape[2])
    return a+'$\\times$'+b+'$\\times$'+c+' $\mu$m${}^3$'

