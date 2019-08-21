import vtk
from vtk.util import numpy_support
import numpy as np
from polaris2.geomvis import utilmpl, utilvtk, utilsh
import logging
log = logging.getLogger('log')

# List of dipoles
# List of positions in data_xyz = [[x0,y0,z0],...,[xN,yN,zN] form
# List of radii in data_j = [[j0,...,jM]_0,...,[j0,...,jM]_N] form where
# the each entry is the radius at a fibonacci_sphere point.
#
# Single directions can be data_j = [[sx0,sy0,sz0],...,[sxN,syN,szN]] form.
class xyzj_list:
    def __init__(self, data_xyz, data_j, shape=[10,10,2.5], xlabel='', title=''):
        
        self.data_xyz = np.array(data_xyz)
        self.data_j = np.array(data_j)
        self.M = self.data_xyz.shape[0] 
        self.shape = shape # um

        self.xlabel = xlabel
        self.title = title

        # Setup renderer
        self.ren, self.renWin, self.iren = utilvtk.setup_render()

    def build_actors(self):
        # Add double arrows for single dipole
        if self.data_j.shape[1] == 3:
            for i in range(self.M):
                utilvtk.draw_double_arrow(self.ren, self.data_xyz[i], self.data_j[i])
        # Add spherical function for ensemble
        else:
            utilvtk.draw_sphere_field(self.ren, self.data_xyz, self.data_j)

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

    def to_xyzJ_list(self, lmax=4):
        # For ensembles only
        N = self.data_j.shape[-1]
        J = utilsh.maxl2maxj(lmax)
        B = utilsh.calcB(N, J)
        data_J = np.einsum('ij,ki->kj', B, self.data_j)
        return xyzJ_list(self.data_xyz, data_J, shape=self.shape,
                         xlabel=self.xlabel, title=self.title)

# Dipole distribution at a single position in the form
# [x0,y0,z0,[J0, J1, ..., JN] where [J0, ..., JN] are even spherical harmonic
# coefficients. 
class xyzJ_list:
    def __init__(self, data_xyz, data_J, shape=[10,10,4], N=2**12, xlabel='', title=''):
        self.data_xyz = np.array(data_xyz)
        self.data_J = np.array(data_J)
        self.M = self.data_xyz.shape[0] 
        self.N = N
        self.shape = shape # um

        self.xlabel = xlabel
        self.title = title

        # Setup renderer
        self.ren, self.renWin, self.iren = utilvtk.setup_render()

        # Calculate dimensions
        self.lmax, mm = utilsh.j2lm(self.data_J.shape[-1] - 1)
        self.J = utilsh.maxl2maxj(self.lmax)

        # Fill the rest of the last l band with zeros
        if self.data_J.shape[-1] != self.J:
            temp = np.zeros(self.J)
            temp[:self.data_J.shape[-1]] = np.array(self.data_J)
            self.data_J = temp

        # Calc points for spherical plotting
        self.xyz = utilsh.fibonacci_sphere(N, xyz=True)
        self.B = utilsh.calcB(self.N, self.J)

    def build_actors(self):
        # Plots spheres
        radii = np.einsum('ij,kj->ki', self.B, self.data_J)
        radii /= np.max(radii)
        utilvtk.draw_sphere_field(self.ren, self.data_xyz, radii)
        
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
        ijk = np.array(np.nonzero(mask)).T
        J_list = self.data[ijk[:,0], ijk[:,1], ijk[:,2], :]
        
        log.info('Plotting '+str(ijk.shape[0])+' ODFs.')
        
        # Draw odfs
        centers = (ijk - 0.5*self.npx)*self.vox_dims # ijk2xyz
        radii = np.einsum('ij,kj->ki', self.B, J_list) 
        utilvtk.draw_sphere_field(self.ren, centers, radii)

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
        
