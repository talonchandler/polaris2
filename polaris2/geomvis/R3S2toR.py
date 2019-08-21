import vtk
from vtk.util import numpy_support
import numpy as np
from polaris2.geomvis import utilmpl, utilvtk, utilsh
import logging
log = logging.getLogger('log')

# Single dipole in [x0,y0,z0,sx0,sy0,sz0] form or
# distribution of dipoles in [x0,y0,z0,[j0, j1, ..., jN]] form
# where sx0,sy0,sz0 are on the units sphere and [j0,...,jN] are radii at
# the fibonacci_sphere points. 
class xyzj_single:
    def __init__(self, data, shape=[10,10,2.5], xlabel='', title=''):
        self.data = data
        self.shape = shape # um

        self.xlabel = xlabel
        self.title = title

        # Setup renderer
        self.ren, self.renWin, self.iren = utilvtk.setup_render()

    def build_actors(self):
        # Add double arrows for single dipole
        if len(self.data) == 6:
            utilvtk.draw_double_arrow(self.ren, *self.data)
        # Add spherical function for ensemble
        else:
            radii = self.data[-1]
            xyz = utilsh.fibonacci_sphere(radii.shape[0], xyz=True)
            pradii = radii.clip(min=0)/np.max(np.abs(radii))
            nradii = -radii.clip(max=0)/np.max(np.abs(radii))
            utilvtk.draw_sphere_function(self.ren, xyz, np.array(self.data[0:3]), pradii, nradii)
        
        utilvtk.draw_origin_dot(self.ren)
        utilvtk.draw_outer_box(self.ren, *self.shape)
        utilvtk.draw_axes(self.ren, *self.shape)
        
        # Set cameras
        dist = 1.1*np.linalg.norm(self.shape)
        self.ren.GetActiveCamera().SetPosition(np.array([1,-1,1])*dist)
        self.ren.GetActiveCamera().SetViewUp([0,0,1])

    def increment_camera(self, az):
        self.ren.GetActiveCamera().Azimuth(az)

    def plot(self, f, fc):
        ax = utilmpl.plot_template(f, fc, xlabel=self.xlabel, title=self.title,
                                scale_bar=False, bump=1.2)

        utilvtk.vtk2imshow(self.renWin, ax[0])
        ax[0].axis('off')
        ax[1].axis('off')

    def to_xyzJ_single(self, lmax=4):
        # For ensembles only
        N = self.data[-1].shape[0]
        J = utilsh.maxl2maxj(lmax)
        B = utilsh.calcB(N, J)
        dataJ = np.einsum('ij,i->j', B, self.data[-1])
        return xyzJ_single([self.data[0], self.data[1], self.data[2], dataJ],
                           shape=self.shape, xlabel=self.xlabel, title=self.title)

# Dipole distribution at a single position in the form
# [x0,y0,z0,[J0, J1, ..., JN] where [J0, ..., JN] are even spherical harmonic
# coefficients. 
class xyzJ_single:
    def __init__(self, data, shape=[10,10,4], N=2**12, xlabel='', title=''):
        self.data = data
        self.N = N
        self.shape = shape # um

        self.xlabel = xlabel
        self.title = title

        # Setup renderer
        self.ren, self.renWin, self.iren = utilvtk.setup_render()

        # Calculate dimensions
        self.lmax, mm = utilsh.j2lm(len(self.data[-1]) - 1)
        self.J = utilsh.maxl2maxj(self.lmax)

        # Fill the rest of the last l band with zeros
        if len(self.data[-1]) != self.J:
            temp = np.zeros(self.J)
            temp[:len(self.data[-1])] = np.array(self.data[-1])
            self.data[-1] = temp
        else:
            self.data[-1] = np.array(data[-1])

        # Calc points for spherical plotting
        self.xyz = utilsh.fibonacci_sphere(N, xyz=True)
        self.B = utilsh.calcB(self.N, self.J)

    def build_actors(self):
        # Calculate positive and negative lobes
        radii = np.einsum('ij,j->i', self.B, self.data[-1])
        pradii = radii.clip(min=0)/np.max(np.abs(radii))
        nradii = -radii.clip(max=0)/np.max(np.abs(radii))
        utilvtk.draw_sphere_function(self.ren, self.xyz, np.array(self.data[0:3]), pradii, nradii)

        # Draw extras
        utilvtk.draw_origin_dot(self.ren)
        utilvtk.draw_outer_box(self.ren, *self.shape)
        utilvtk.draw_axes(self.ren, *self.shape)
        
        # Set cameras
        dist = 1.1*np.linalg.norm(self.shape)
        self.ren.GetActiveCamera().SetPosition(np.array([1,-1,1])*dist)
        self.ren.GetActiveCamera().SetViewUp([0,0,1])

    def increment_camera(self, az):
        self.ren.GetActiveCamera().Azimuth(az)

    def plot(self, f, fc):
        ax = utilmpl.plot_template(f, fc, xlabel=self.xlabel, title=self.title,
                                scale_bar=False, bump=1.2)

        utilvtk.vtk2imshow(self.renWin, ax[0])
        ax[0].axis('off')
        ax[1].axis('off')

# Dipole distribution at very position in a volume.
# data is a 4D array with xyz positions on the first three dimensions and
# real even spherical harmonic coefficients on the last dimension.
class xyzJ:
    def __init__(self, data, vox_dims=[.1,.1,.1], N=2**10, xlabel='', title='',
                 max_in_rad=1.0, max_out_rad=1.0):
        self.data = data
        self.npx = np.array(self.data.shape[0:3])
        self.vox_dims = vox_dims # um
        self.N = N
        self.shape = np.array(data.shape[0:3])*np.array(vox_dims)

        self.xlabel = xlabel
        self.title = title
        self.max_in_rad = max_in_rad
        self.max_out_rad = max_out_rad

        # Setup renderer
        self.ren, self.renWin, self.iren = utilvtk.setup_render()

        # Calculate dimensions
        self.lmax, mm = utilsh.j2lm(self.data.shape[-1] - 1)
        self.J = utilsh.maxl2maxj(self.lmax)

        # # Fill the rest of the last l band with zeros
        # if len(self.data.shape[-1]) != self.J:
        #     temp = np.zeros(self.J)
        #     temp[:len(self.data.shape[-1])] = np.array(self.data.shape[-1])
        #     self.data[-1] = temp
        # else:
        #     self.data[-1] = np.array(data[-1])

        # Calc points for spherical plotting
        self.xyz = utilsh.fibonacci_sphere(self.N, xyz=True)
        self.B = utilsh.calcB(self.N, self.J)

    def build_actors(self):
        mask = self.data[:,:,:,0] > 0
        odf_ijk_list = np.array(np.nonzero(mask)).T
        log.info('Plotting '+str(odf_ijk_list.shape[0])+' ODFs.')
        
        # Plot each odf
        for odf_ijk in odf_ijk_list:
            J = self.data[odf_ijk[0], odf_ijk[1], odf_ijk[2], :]
            # Calculate positive and negative lobes
            radii = np.einsum('ij,j->i', self.B, J)
            
            # Scale input if too large
            max_rad = np.max(radii)
            if max_rad > self.max_in_rad:
                radii *= self.max_in_rad/max_rad

            # Scale output to maximum radius
            radii *= self.max_out_rad/self.max_in_rad
            
            pradii = radii.clip(min=0)
            nradii = -radii.clip(max=0)
            
            center = (odf_ijk - 0.5*self.npx)*self.vox_dims # ijk2xyz
            utilvtk.draw_sphere_function(self.ren, self.xyz, center, pradii, nradii)

        # Draw extras
        utilvtk.draw_outer_box(self.ren, *self.shape)
        utilvtk.draw_axes(self.ren, *self.shape)
        
        # Set cameras
        dist = 1.1*np.linalg.norm(self.shape)
        self.ren.GetActiveCamera().SetPosition(np.array([1,-1,1])*dist)
        self.ren.GetActiveCamera().SetViewUp([0,0,1])

    def increment_camera(self, az):
        self.ren.GetActiveCamera().Azimuth(az)

    def plot(self, f, fc):
        ax = utilmpl.plot_template(f, fc, xlabel=self.xlabel, title=self.title,
                                scale_bar=False, bump=1.2)

        utilvtk.vtk2imshow(self.renWin, ax[0])
        ax[0].axis('off')
        ax[1].axis('off')
        
