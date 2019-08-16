import vtk
from vtk.util.numpy_support import vtk_to_numpy
from scipy.special import sph_harm
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
def plot(input_objects, output_file='output.pdf', dpi=300):
    mkdir(output_file)
    
    rows = len(input_objects)
    cols = len(input_objects[0])

    f = fig.Figure(figsize=(3*cols,3*rows))
    for row in range(rows):
        for col in range(cols):
            fc = [col/cols, (rows-row-1)/rows, 1/cols, 1/rows]
            input_objects[row][col].plot(f, fc)
            
    f.savefig(output_file, dpi=dpi)

# Plot template for 2D and 3D matplotlib geometries
def plot_template(f, fc, shape=(1,1), xlabel='', ylabel='', clabel='', title='',
                  scale_bar=True):
    # Subfigure coords
    fx, fy, fw, fh = fc

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

    # Center coordinates
    cx = fx + 0.425*fw
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
    if scale_bar:
        scale_shift = 0.05*fh
        if len(shape) == 2:
            ax0.annotate('', xy=(cx-w,cy-h-scale_shift), xytext=(cx+w, cy-h-scale_shift), xycoords='figure fraction', textcoords='figure fraction', va='center', arrowprops=dict(arrowstyle="|-|, widthA=0.5, widthB=0.5", shrinkA=0, shrinkB = 0, lw=.75))
            ax0.annotate(xlabel, xy=(1,1), xytext=(cx,cy-h-0.1*fh), textcoords='figure fraction', ha='center', va='center', rotation=0)
        elif len(shape) == 3:
            ax0.annotate('', xy=(acx+sw,cy-h-scale_shift), xytext=(cx+w, cy-h-scale_shift), xycoords='figure fraction', textcoords='figure fraction', va='center', arrowprops=dict(arrowstyle="|-|, widthA=0.5, widthB=0.5", shrinkA=0, shrinkB = 0, lw=.75))            
            ax0.annotate(xlabel, xy=(1,1), xytext=(acx + (cx+w - acx)/2,cy-h-0.1*fh), textcoords='figure fraction', ha='center', va='center', rotation=0)

    # Labels
    ax0.annotate(title, xy=(1,1), xytext=(cx,cy+hnom+0.05*fh), textcoords='figure fraction', ha='center', va='center')
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

# Setup vtk render and renderwindow
def setup_render(size=2000):
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetSize(size, size)
    ren.SetBackground([255,255,255])
    renWin.OffScreenRenderingOff()
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    return ren, renWin, iren

# Plots a vtk renderWindow and in a matplotlib axis with imshow
def vtk2imshow(renWin, ax):
    # Render
    renWin.OffScreenRenderingOn()
    renWin.Render()

    # Filter renWin
    image_filter = vtk.vtkWindowToImageFilter()
    image_filter.SetInput(renWin)
    image_filter.SetScale(1)
    image_filter.SetInputBufferTypeToRGB()
    image_filter.ReadFrontBufferOff()
    image_filter.Update()

    # Convert to numpy array 
    im = image_filter.GetOutput()
    rows, cols, _ = im.GetDimensions()
    sc = im.GetPointData().GetScalars()
    a = vtk_to_numpy(sc)
    a = a.reshape(rows, cols, -1)
    ax.imshow(a, origin='lower')

# Make axes acotr for vtk plotting
def make_axes():
    axes = vtk.vtkAxesActor()
    axes.SetShaftTypeToCylinder()
    axes.SetXAxisLabelText('')
    axes.SetYAxisLabelText('')
    axes.SetZAxisLabelText('')
    axes.SetTotalLength(1.5, 1.5, 1.5)
    axes.SetCylinderRadius(0.75 * axes.GetCylinderRadius())
    axes.SetConeRadius(1.5 * axes.GetConeRadius())
    axes.SetSphereRadius(1.5 * axes.GetSphereRadius())
    return axes

# SciPy real spherical harmonics with identical interface to SymPy's Znm
# Useful for fast numerical evaluation of Znm
def spZnm(l, m, theta, phi):
    if m > 0:
        return np.sqrt(2)*((-1)**m)*np.real(sph_harm(m, l, phi, theta))
    elif m == 0:
        return np.real(sph_harm(m, l, phi, theta))
    elif m < 0:
        return np.sqrt(2)*((-1)**m)*np.imag(sph_harm(np.abs(m), l, phi, theta))

# Convert between spherical harmonic indices (l, m) and multi-index (j)
def j2lm(j):
    if j < 0:
        return None
    l = 0
    while True:
        x = 0.5*l*(l+1)
        if abs(j - x) <= l:
            return l, int(j-x)
        else:
            l = l+2

def lm2j(l, m):
    if abs(m) > l or l%2 == 1:
        return None
    else:
        return int(0.5*l*(l+1) + m)

def maxl2maxj(l):
    return int(0.5*(l + 1)*(l + 2))

# Convert between Cartesian and spherical coordinates
def tp2xyz(tp):
    theta = tp[0]
    phi = tp[1]
    return np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)

def xyz2tp(x, y, z):
    arccos_arg = z/np.sqrt(x**2 + y**2 + z**2)
    if np.isclose(arccos_arg, 1.0): # Avoid arccos floating point issues
        arccos_arg = 1.0
    elif np.isclose(arccos_arg, -1.0):
        arccos_arg = -1.0
    return np.arccos(arccos_arg), np.arctan2(y, x)

# Returns "equally" spaced points on a unit sphere in spherical coordinates.
# http://stackoverflow.com/a/26127012/5854689
def fibonacci_sphere(n, xyz=False):
    z = np.linspace(1 - 1/n, -1 + 1/n, num=n) 
    theta = np.arccos(z)
    phi = np.mod((np.pi*(3.0 - np.sqrt(5.0)))*np.arange(n), 2*np.pi) - np.pi
    if xyz:
        return np.vstack((np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta))).T
    else:
        return np.vstack((theta, phi)).T
