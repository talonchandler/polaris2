import vtk
from vtk.util import numpy_support
import numpy as np
from polaris2.geomvis import R3toR3, R3toR, utilmpl, utilvtk, utilsh
import tifffile
import logging
log = logging.getLogger('log')

# List of dipoles
# List of positions in data_xyz = [[x0,y0,z0],...,[xM,yM,zM] form
# List of radii in data_j = [[j0,...,jN]_0,...,[j0,...,jN]_M] form where
# the each entry is the radius at a fibonacci_sphere point.
#
# Single directions can be data_j = [[sx0,sy0,sz0],...,[sxM,syM,szM]] form.
class xyzj_list:
    def __init__(self, data_xyz, data_j, shape=[10,10,2.5], title='',
                 rad_scale=1):
        
        self.data_xyz = np.array(data_xyz)
        self.data_j = np.array(data_j)
        self.M = self.data_xyz.shape[0] 
        self.shape = shape # um

        self.xlabel = utilmpl.shape2xlabel(self.shape)
        self.title = title
        self.rad_scale = rad_scale

    def build_actors(self):

        log.info('Plotting '+str(self.data_xyz.shape[0])+' ODFs.')
        
        # Setup renderer
        self.ren, self.renWin, self.iren = utilvtk.setup_render()
        
        # Add double arrows for single dipole
        if self.data_j.shape[1] == 3:
            for i in range(self.M):
                utilvtk.draw_double_arrow(self.ren, self.data_xyz[i], self.rad_scale*self.data_j[i])
        # Add spherical function for ensemble
        else:
            radii = self.data_j*self.rad_scale/np.max(self.data_j)
            utilvtk.draw_sphere_field(self.ren, self.data_xyz, radii)

        # Draw extras
        utilvtk.draw_origin_dot(self.ren)
        utilvtk.draw_outer_box(self.ren, *self.shape)
        utilvtk.draw_axes(self.ren, *self.shape)
        
        # Set cameras
        dist = 1.15*np.linalg.norm(self.shape)
        self.ren.GetActiveCamera().SetPosition(np.array([1,-1,1])*dist)
        self.ren.GetActiveCamera().SetViewUp([0,0,1])

    def increment_camera(self, az):
        self.ren.GetActiveCamera().Azimuth(az)

    def plot(self, f, fc, ss):
        ax = utilmpl.plot_template(f, fc, xlabel=self.xlabel, title=self.title,
                                scale_bar=False, bump=1.2)

        utilvtk.vtk2imshow(self.renWin, ax[0], ss)
        ax[0].axis('off')
        ax[1].axis('off')

    def to_xyzJ_list(self, Jmax=15):
        if self.data_j.shape[1] == 3: # For single dipoles
            data_J = np.zeros((1, Jmax))
            t, p = utilsh.xyz2tp(*self.data_j[0])
            for i in range(Jmax):
                l, m = utilsh.j2lm(i)
                data_J[0,i] = utilsh.spZnm(l, m, t, p)
        else: # For ensembles
            N = self.data_j.shape[-1]
            B = utilsh.calcB(N, Jmax)
            data_J = np.einsum('ij,ki->kj', B, self.data_j)
        return xyzJ_list(self.data_xyz, data_J, shape=self.shape, title=self.title)

    def to_R3toR3_xyz(self, N=2**10):
        xyz = utilsh.fibonacci_sphere(N, xyz=True)
        max_indices = np.argmax(self.data_j, axis=1)
        xyz_max = np.einsum('ij,i->ij', xyz[max_indices], np.max(self.data_j, axis=-1))
        return R3toR3.xyz_list(self.data_xyz, xyz_max, shape=self.shape,
                               xlabel=self.xlabel, title='Peaks')

    def to_xyzJ(self, xyzJ_shape=[10,10,10,6], vox_dims=[.1,.1,.1]):        
        xyzJ_list = self.to_xyzJ_list(Jmax=xyzJ_shape[-1])
        return xyzJ_list.to_xyzJ(xyzJ_shape=xyzJ_shape, vox_dims=vox_dims)
        
# Dipole distribution at a single position in the form
# [x0,y0,z0,[J0, J1, ..., JN] where [J0, ..., JN] are even spherical harmonic
# coefficients. 
class xyzJ_list:
    def __init__(self, data_xyz, data_J, shape=[10,10,4], N=2**12, title=''):
        self.data_xyz = np.array(data_xyz)
        self.data_J = np.array(data_J)
        self.M = self.data_xyz.shape[0] 
        self.N = N
        self.shape = shape # um

        self.xlabel = utilmpl.shape2xlabel(self.shape)
        self.title = title

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
        log.info('Plotting '+str(self.data_xyz.shape[0])+' ODFs.')

        # Setup renderer
        self.ren, self.renWin, self.iren = utilvtk.setup_render()

        # Plots spheres
        radii = np.einsum('ij,kj->ki', self.B, self.data_J)
        radii /= np.max(radii)
        utilvtk.draw_sphere_field(self.ren, self.data_xyz, radii)
        
        # Draw extras
        utilvtk.draw_origin_dot(self.ren)
        utilvtk.draw_outer_box(self.ren, *self.shape)
        utilvtk.draw_axes(self.ren, *self.shape)
        
        # Set cameras
        dist = 1.15*np.linalg.norm(self.shape)
        self.ren.GetActiveCamera().SetPosition(np.array([1,-1,1])*dist)
        self.ren.GetActiveCamera().SetViewUp([0,0,1])

    def increment_camera(self, az):
        self.ren.GetActiveCamera().Azimuth(az)

    def plot(self, f, fc, ss):
        ax = utilmpl.plot_template(f, fc, xlabel=self.xlabel, title=self.title,
                                scale_bar=False, bump=1.2)

        utilvtk.vtk2imshow(self.renWin, ax[0], ss)
        ax[0].axis('off')
        ax[1].axis('off')

    def to_xyzj_list(self, N):
        J = utilsh.maxl2maxj(self.lmax)
        B = utilsh.calcB(N, self.J)
        data_j = np.einsum('ij,kj->ki', B, self.data_J)
        return xyzj_list(self.data_xyz, data_j, shape=self.shape,
                         title=self.title)

    def to_xyzJ(self, xyzJ_shape=[10,10,10,6], vox_dims=[.1,.1,.1]):
        out = np.zeros(xyzJ_shape)
        npx = xyzJ_shape[0:3]
        ijk_count = np.floor(self.data_xyz/vox_dims + (np.array(npx)/2)).astype(np.int)
        for m, ijk in enumerate(ijk_count):
            out[ijk[0], ijk[1], ijk[2], :] += self.data_J[m,:xyzJ_shape[3]]

        return xyzJ(out, vox_dims=vox_dims, title=self.title)

# Dipole distribution at very position in a volume.
# data is a 4D array with xyz positions on the first three dimensions and
# real even spherical harmonic coefficients on the last dimension.
class xyzJ:
    def __init__(self, data, vox_dims=[.1,.1,.1], N=2**10, title='',
                 skip_n=1, rad_scale=1, threshold=0):
        self.data = data
        self.npx = np.array(self.data.shape[0:3])
        self.vox_dims = vox_dims # um
        self.N = N
        self.shape = np.array(data.shape[0:3])*np.array(vox_dims)

        self.xlabel = utilmpl.shape2xlabel(self.shape)
        self.title = title
        self.skip_n = skip_n
        self.rad_scale = rad_scale
        self.threshold = threshold

        # Calculate dimensions
        self.lmax, mm = utilsh.j2lm(self.data.shape[-1] - 1)
        self.J = utilsh.maxl2maxj(self.lmax)

    def build_actors(self):
        # Setup renderer
        self.ren, self.renWin, self.iren = utilvtk.setup_render()
        
        self.B = utilsh.calcB(self.N, self.J)
        
        thresh_mask = self.data[:,:,:,0] > self.threshold
        skip_mask = np.zeros_like(thresh_mask, dtype=np.bool)
        skip_mask[::self.skip_n,::self.skip_n,::self.skip_n] = 1
        
        ijk = np.array(np.nonzero(thresh_mask*skip_mask)).T
        J_list = self.data[ijk[:,0], ijk[:,1], ijk[:,2], :]
        
        log.info('Plotting '+str(ijk.shape[0])+' ODFs.')

        # Draw odfs
        centers = (ijk - 0.5*self.npx + 0.5)*self.vox_dims # ijk2xyz
        radii = np.einsum('ij,kj->ki', self.B, J_list)
        radii *= self.rad_scale*self.skip_n*np.min(self.vox_dims)/(np.max(radii))
        utilvtk.draw_sphere_field(self.ren, centers, radii)

        # Draw extras
        utilvtk.draw_origin_dot(self.ren)
        utilvtk.draw_outer_box(self.ren, *self.shape)
        utilvtk.draw_axes(self.ren, *self.shape)
        
        # Set cameras
        dist = 1.15*np.linalg.norm(self.shape)
        self.ren.GetActiveCamera().SetPosition(np.array([1,-1,1])*dist)
        self.ren.GetActiveCamera().SetViewUp([0,0,1])

    def increment_camera(self, az):
        self.ren.GetActiveCamera().Azimuth(az)

    def plot(self, f, fc, ss):
        ax = utilmpl.plot_template(f, fc, xlabel=self.xlabel, title=self.title,
                                scale_bar=False, bump=1.2)

        utilvtk.vtk2imshow(self.renWin, ax[0], ss)
        ax[0].axis('off')
        ax[1].axis('off')

    def to_xyzJ_list(self):
        thresh_mask = self.data[:,:,:,0] > self.threshold
        skip_mask = np.zeros_like(thresh_mask, dtype=np.bool)
        skip_mask[::self.skip_n,::self.skip_n,::self.skip_n] = 1
        
        ijk = np.array(np.nonzero(thresh_mask*skip_mask)).T
        centers = (ijk - 0.5*self.npx +  + 0.5)*self.vox_dims # ijk2xyz        
        J_list = self.data[ijk[:,0], ijk[:,1], ijk[:,2], :]

        return xyzJ_list(centers, J_list, title=self.title, shape=self.shape)

    def to_R3toR3_xyz(self, N=2**10):
        xyzJ_list = self.to_xyzJ_list()
        xyzj_list = xyzJ_list.to_xyzj_list(N)
        return xyzj_list.to_R3toR3_xyz()

    def to_R3toR_xyz(self):
        return R3toR.xyz(self.data[:,:,:,0],
                         vox_dims=self.vox_dims, 
                         title='Maximum intensity projection')

    def to_tiff(self, filename):
        utilmpl.mkdir(filename)
        log.info('Writing '+filename)
        
        with tifffile.TiffWriter(filename, imagej=True) as tif:
            d = np.moveaxis(self.data, [2, 3, 1, 0], [0, 1, 2, 3]).astype(np.float32)
            tif.save(d[None,:,:,:,:],
                     resolution=(1/self.vox_dims[0], 1/self.vox_dims[1]),
                     metadata={'spacing': self.vox_dims[2], 'unit':'um'}) # TZCYXS

    def from_tiff(self, filename):
        log.info('Reading '+filename)
        with tifffile.TiffFile(filename) as tf:
            # Read data
            self.data = np.ascontiguousarray(np.moveaxis(tf.asarray(), [0, 1, 2, 3], [2, 3, 1, 0]))

            # Read vox_dims from metadata
            xx = tf.pages[0].tags['XResolution'].value
            self.vox_dims[0] = xx[1]/xx[0]
            yy = tf.pages[0].tags['YResolution'].value
            self.vox_dims[1] = yy[1]/yy[0]
            self.vox_dims[2] = tf.imagej_metadata['spacing']
            
        self.npx = np.array(self.data.shape[0:3])
        self.shape = np.array(self.npx)*np.array(self.vox_dims)
        self.xlabel = utilmpl.shape2xlabel(self.shape)
        self.lmax, mm = utilsh.j2lm(self.data.shape[-1] - 1)
        self.J = utilsh.maxl2maxj(self.lmax)
