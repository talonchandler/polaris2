from tqdm import trange, tqdm
import numpy as np
from polaris2.geomvis import R2toR, R2toC2, R3S2toR, utilsh
import logging
log = logging.getLogger('log')

# Simulates a linear dipole imaged by 4f detection system.
class FourF:

    def __init__(self, NA=1.2, M=60, n0=1.3, lamb=0.546, wpx_real=6.5,
                 npx=(7*17, 7*17), ss=2, plotfov=10, irrad_title='4$f$ detector irradiance'):
        # Input parameters
        self.NA = NA
        self.M = M # magnification
        self.n0 = n0 # object index of refraction
        self.lamb = lamb # wavelength 
        self.npx = npx # number of pixels on detector
        self.ss = ss # super-sample factor for each pixel
        self.wpx_real = wpx_real # pixel width (from camera data sheet)

        # Plotting parameters
        self.plotfov = plotfov # detctor region plotted
        self.irrad_title = irrad_title
        
        # Derived parameters
        self.npxss = npx[0]*ss
        self.wpx = self.wpx_real/M # pixel width in object space
        self.fov = self.wpx*self.npx[0] # demagnified FOV        
        self.nuc = 2*NA/lamb # transverse cutoff frequency
        self.num = n0/lamb # Ewald sphere radius
        self.wbfp = 1/(self.wpx*self.npx[0]) # sample spacing in BFP

        # FOV in bfp
        if self.npxss % 2 == 0:
            self.bfpmin = -0.5*self.npxss*self.wbfp
            self.bfpmax = (0.5*self.npxss-1)*self.wbfp
        else:
            self.bfpmin = -0.5*(self.npxss-1)*self.wbfp
            self.bfpmax = 0.5*(self.npxss-1)*self.wbfp

    # ----------
    # To back focal plane
            
    # Generates the electric field pattern in the bfp due to a single dipole
    # Input: R3S2toR.xyzj_list with a single entry
    # Output: R2toC2.xy object
    # Based on: Backer, A. S., & Moerner, W. E. (2014)
    # http://dx.doi.org/10.1021/jp501778z
    def xyzj_single_to_xye_bfp(self, dip):
        dip_pos = dip.data_xyz[0]
        dip_orientation = dip.data_j[0]
        
        # Precompute self.h_xyzJ_single_to_xye
        self.precompute_xyzJ_single_to_xye_bfp(dip_pos)
        
        # Matrix multiplication
        out = np.einsum('ijkl,l->ijk', self.h_xyzJ_single_to_xye_bfp, dip_orientation)

        # Return BFP
        return R2toC2.xy(out,
                         circle=True, title='Scaled back focal plane fields',
                         xlabel='$\\textrm{2NA}/\lambda$',
                         toplabel='$E_x$', bottomlabel='$E_y$',
                         fov=[self.bfpmin, self.bfpmax],
                         plotfov=[-self.nuc/2, self.nuc/2],
                         colormax=1.5)
    
    def precompute_xyzJ_single_to_xye_bfp(self, dip_pos):
        rx, ry, rz = dip_pos
        
        # Coordinates
        x = np.linspace(self.bfpmin, self.bfpmax, self.npxss)
        y = np.linspace(self.bfpmin, self.bfpmax, self.npxss)

        taux, tauy = np.meshgrid(x, y, indexing='ij')
        tauphi = np.arctan2(tauy, taux)
        abstau = np.sqrt(taux**2 + tauy**2)
        rho = abstau/self.num

        # Apodization and phase calculations
        apod = np.where(abstau < self.nuc/2, 1, 0)
        sine_apod = (1 - apod*rho**2)**(-0.25)
        sqrtrho = np.sqrt(1 - apod*rho**2)
        tphase = np.exp(1j*2*np.pi*(taux*rx + tauy*ry))
        aphase = np.exp(1j*2*np.pi*rz*np.sqrt(self.num**2 - apod*abstau**2))
        pre = apod*sine_apod*tphase*aphase
        
        # Compute matrix elements
        self.h_xyzJ_single_to_xye_bfp = np.zeros((self.npxss, self.npxss, 2, 3), dtype='complex64')
        self.h_xyzJ_single_to_xye_bfp[:,:,0,0] = (np.sin(tauphi))**2 + (np.cos(tauphi))**2*sqrtrho # gxx
        self.h_xyzJ_single_to_xye_bfp[:,:,0,1] = 0.5*np.sin(2*tauphi)*(sqrtrho - 1) # gxy
        self.h_xyzJ_single_to_xye_bfp[:,:,0,2] = rho*np.cos(tauphi) # gxz
        self.h_xyzJ_single_to_xye_bfp[:,:,1,0] = self.h_xyzJ_single_to_xye_bfp[:,:,0,1] # gxy
        self.h_xyzJ_single_to_xye_bfp[:,:,1,1] = (np.cos(tauphi))**2 + (np.sin(tauphi))**2*sqrtrho # gyy
        self.h_xyzJ_single_to_xye_bfp[:,:,1,2] = rho*np.sin(tauphi) # gyz

        self.h_xyzJ_single_to_xye_bfp = np.einsum('ijkl,ij->ijkl', self.h_xyzJ_single_to_xye_bfp, pre)

        # TODO: Add monopole option flag
        # # Monopole 
        # if sx == 0 and sy == 0 and sz == 0:
        #     self.h_xyzJ_single_to_xye_bfp = np.einsum('ijkl,ij->ijkl', self.h_xyzJ_single_to_xye_bfp, pre)

    # ----------
    # To detector
            
    # Propagates the electric fields from the bfp to the image plane 
    # Input: R2toC2.xy object
    # Output: R2toC2.xy object
    def xye_bfp_to_xye_det(self, ebfp):
        # Apply phase shift for even self.npx
        if self.npxss % 2 == 0:
            x = np.arange(-self.npxss//2, self.npxss//2)
            y = np.arange(-self.npxss//2, self.npxss//2)
            xx, yy = np.meshgrid(x, y)
            phase_ramp = np.exp(-1j*np.pi*(xx + yy)/self.npxss)
            to_ft = np.einsum('ijk,ij->ijk', ebfp.data, phase_ramp)
        else:
            to_ft = ebfp.data

        # Fourier transforms
        shifted = np.fft.ifftshift(to_ft, axes=(0,1))
        ffted = np.fft.fft2(shifted, axes=(0,1))*(self.wbfp**2)
        result = np.fft.fftshift(ffted, axes=(0,1))

        return R2toC2.xy(result,
                         fov=[-self.fov/2, self.fov/2],
                         plotfov=[-self.plotfov/2, self.plotfov/2],
                         title='Image plane fields',
                         xlabel=str(self.plotfov)+' $\mu$m',
                         toplabel='$E_x$', bottomlabel='$E_y$')

    # Generates the electric field pattern on the detector due to a single dipole
    # Input: R3S2toR.xyzj_list with a single entry
    # Output: R2toC2.xy object
    def xyzj_single_to_xye_det(self, dip):
        dip_pos = dip.data_xyz[0]
        dip_orientation = dip.data_j[0]
        
        # Precompute self.h_xyzJ_single_to_xye
        self.precompute_xyzJ_single_to_xye_det(dip_pos)
        
        # Matrix multiplication
        out = np.einsum('ijkl,l->ijk', self.h_xyzJ_single_to_xye_det, dip_orientation)

        return R2toC2.xy(out,
                         fov=[-self.fov/2, self.fov/2],
                         plotfov=[-self.plotfov/2, self.plotfov/2],
                         title='Image plane fields',
                         xlabel=str(self.plotfov)+' $\mu$m',
                         toplabel='$E_x$', bottomlabel='$E_y$')

    def precompute_xyzJ_single_to_xye_det(self, dip):
        self.precompute_xyzJ_single_to_xye_bfp(dip)

        self.h_xyzJ_single_to_xye_det = np.zeros((self.npxss, self.npxss, 2, 3), dtype='complex64')
        for i in range(3):
            temp_bfp = R2toC2.xy(self.h_xyzJ_single_to_xye_bfp[...,i])
            self.h_xyzJ_single_to_xye_det[...,i] = self.xye_bfp_to_xye_det(temp_bfp).data

    # Generates irradiance from electric fields
    # Input: R2toC2.xy object
    # Output: R2toR.xy object
    def xye_to_xy_det(self, e):
        # Calculate irradiance
        irr = np.sum(np.abs(e.data)**2, axis=-1) 

        # "Undo" supersampling by summing over squares
        irrpx = irr.reshape(self.npx[0], self.ss, self.npx[1], self.ss).sum(axis=(1,3))

        # Return 
        return R2toR.xy(irrpx, 
                        title=self.irrad_title,
                        fov=[-self.fov/2, self.fov/2],
                        plotfov=[-self.plotfov/2, self.plotfov/2])

    # Generates the irradiance pattern due to a single dipole
    # This is an abstraction over the main functions in this class
    # Input: R3S2toR.xyzj_single with a single dipole
    # Output: R2toC2.xy object
    def xyzj_single_to_xy_det(self, dip): # dip_to_det
        eim = self.xyzj_single_to_xye_det(dip)
        return self.xye_to_xy_det(eim)

    # Generates the irradiance pattern due to a single dipole distribution
    # Input: R3S2toR.xyzJ_list with single entry
    # Output: R2toR.xy object
    def xyzJ_single_to_xy_det(self, dist):
        self.precompute_xyzJ_single_to_xy_det(dist.data_xyz[0])
        out = np.einsum('ijk,k->ij', self.h_xyzJ_single_to_xy_det, dist.data_J[0])

        return R2toR.xy(out, 
                        title=self.irrad_title,
                        fov=[-self.fov/2, self.fov/2],
                        plotfov=[-self.plotfov/2, self.plotfov/2])

    def precompute_xyzJ_single_to_xy_det(self, dist):
        self.precompute_xyzJ_single_to_xye_det(dist)
        self.h_xyzJ_single_to_xy_det = np.zeros((self.npx[0], self.npx[1], 6), dtype='float32')
        
        # Compute gaunt coeffs
        G = utilsh.gaunt_l1l1_tol0l2()

        # Compute matrix
        out = np.real(np.einsum('ijkl,ijkm,nlm->ijn',
                                self.h_xyzJ_single_to_xye_det,
                                self.h_xyzJ_single_to_xye_det.conj(),
                                G)).astype('float32')

        # Downsample and store
        self.h_xyzJ_single_to_xy_det = out.reshape(self.npx[0], self.ss, self.npx[1], self.ss, 6).sum(axis=(1,3))

    # Generates the irradiance pattern due to several dipole distributions
    # This is a slow path for dense objects.
    # Input: R3S2toR.xyzJ_list
    # Output: R2toR.xy object
    def xyzJ_list_to_xy_det(self, dist):
        out = np.zeros(self.npx)
        for m in range(dist.M):
            distm = R3S2toR.xyzJ_list([dist.data_xyz[m]], [dist.data_J[m]])
            out += self.xyzJ_single_to_xy_det(distm).data
        return R2toR.xy(out, 
                        title=self.irrad_title,
                        fov=[-self.fov/2, self.fov/2],
                        plotfov=[-self.plotfov/2, self.plotfov/2])

    # Generates the irradiance pattern due to a dense xyzJ array.
    # This is a faster path than xyzJ_list_to_xy_det
    # Input: R3S2toR.xyzJ
    # Output: R2toR.xy object
    def xyzJ_to_xy_det(self, xyzJ):
        
        # Assume input xy dimensions are <= detector xy dimensions
        if xyzJ.data.shape[0:2] != self.npx:
            log.info('Error! The xy dimensions of the xyzJ object must match the xy detector dimensions.')
            return None
            
        # self.precompute_XYzJ_to_XY_det(xyzJ.data.shape, xyzJ.vox_dims)

        xyzJ_shift = np.fft.ifftshift(xyzJ.data, axes=(0,1))
        XYzJ_shift = np.fft.rfft2(xyzJ_shift, axes=(0,1)) # FT along xy
        XY_shift = np.einsum('ijkl,ijkl->ij', self.H_XYzJ_to_XY, XYzJ_shift) # Filter and sum over J and z

        xy_shift = np.fft.irfft2(XY_shift, s=xyzJ.data.shape[0:2]) # IFT along XY
        xy = np.fft.fftshift(xy_shift)
        
        return R2toR.xy(np.real(xy),
                        px_dims=(self.wpx, self.wpx),
                        title=self.irrad_title,
                        fov=[-self.fov/2, self.fov/2],
                        plotfov=[-self.plotfov/2, self.plotfov/2],
                        xlabel=str(self.plotfov)+' $\mu$m')

    def precompute_XYzJ_to_XY_det(self, input_shape, input_vox_dims):
        # Adjust matrix size to account for rfft
        input_shape_rfft = np.array(input_shape)
        input_shape_rfft[1] = np.floor(input_shape_rfft[1]/2 + 1)
        
        # Fill matrix
        log.info('Computing psfs')
        self.H_XYzJ_to_XY = np.zeros(input_shape_rfft, dtype=np.complex64)
        oddK = (input_shape[2]%2 == 1)
        for k in trange(int(input_shape[2]/2 + 0.5*oddK)):
            z = (k + 0.5*oddK - 0.5*input_shape[2])*input_vox_dims[2] # k2z
            self.precompute_xyzJ_single_to_xy_det([0,0,z])
            h_shift = np.fft.ifftshift(self.h_xyzJ_single_to_xy_det, axes=(0,1))

            # Exploit symmetry above and below the focal plane
            self.H_XYzJ_to_XY[:,:,k,:6] = np.fft.rfft2(h_shift, axes=(0,1))*np.product(input_vox_dims) # For anisometric voxels
            self.H_XYzJ_to_XY[:,:,-k,:6] = self.H_XYzJ_to_XY[:,:,k,:6]
    
# Simulates a linear dipole imaged by 4f system with a microlens array. 
# Depends on FourF class.
class FourFLF:

    def __init__(self, fulen=2500, ulenpx=(2**4 + 1),
                 ulens_aperture='square',
                 **kwargs):
        
        self.fourf = FourF(**kwargs)
        self.fulen = fulen # ulens focal length        
        self.ulenpx = ulenpx # number of pixels behind each ulens
        self.ulens_aperture = ulens_aperture # 'square' or 'circle'

        self.nulen = int(self.fourf.npx[0]/ulenpx)

    # Generates detector fields (after microlenses) due to a single dipole
    # Input: R3S2toR.xyzj_list with a single entry
    # Output: R2toR.xy object
    def xyzj_single_to_xye_det(self, dip):
        dip_pos = dip.data_xyz[0]
        dip_orientation = dip.data_j[0]
        
        # Precompute self.h_xyzJ_single_to_xye
        self.precompute_xyzJ_single_to_xye_det(dip_pos)
        
        # Matrix multiplication
        out = np.einsum('ijkl,l->ijk', self.h_xyzJ_single_to_xye_det, dip_orientation)

        return R2toC2.xy(out,
                         fov=[-self.fourf.fov/2, self.fourf.fov/2],
                         plotfov=[-self.fourf.plotfov/2, self.fourf.plotfov/2],
                         title='Lightfield detector fields',
                         xlabel=str(self.fourf.plotfov)+' $\mu$m',
                         toplabel='$E_x$', bottomlabel='$E_y$')

    def precompute_xyzJ_single_to_xye_det(self, dip_pos):
        self.h_xyzJ_single_to_xye_det = np.zeros((self.fourf.npxss, self.fourf.npxss, 2, 3), dtype='complex64')
        
        # Use FourF to calculate electric field in nominal image plane
        # eim = self.fourf.xyzj_single_to_xye_det(dip)
        self.fourf.precompute_xyzJ_single_to_xye_det(dip_pos)

        # For x, y, z dipoles 
        for i in range(3):
            eim = R2toC2.xy(self.fourf.h_xyzJ_single_to_xye_det[...,i])
            
            # Build microlens tile
            xmin = -0.5*(self.ulenpx-1)*self.fourf.wpx_real
            xmax = 0.5*(self.ulenpx-1)*self.fourf.wpx_real
            x = np.linspace(xmin, xmax, self.ulenpx*self.fourf.ss)
            xx, yy = np.meshgrid(x, x)
            ulentile = np.exp(-1j*np.pi*(xx**2 + yy**2)/(self.fulen*self.fourf.lamb))
            if self.ulens_aperture == 'circle': 
                rr = np.sqrt(xx**2 + yy**2) 
                ulentile *= np.where(rr < xmax, 1, 0)

            # Apply microlens phase            
            tiled = np.tile(ulentile, 2*(self.fourf.npx[0]//self.ulenpx,))
            Eout = np.einsum('ijk,ij->ijk', eim.data, tiled) 

            # Fresnel propagation to detector
            nu = np.fft.fftfreq(self.fourf.npxss, self.fourf.wpx_real/self.fourf.ss)
            nuxx, nuyy = np.meshgrid(nu, nu)
            H = np.exp(-1j*np.pi*self.fourf.lamb*self.fulen*(nuxx**2 + nuyy**2))
            fft2 = np.fft.fft2(np.fft.fftshift(Eout, axes=(0,1)), axes=(0,1))
            filtered = np.einsum('ijk,ij->ijk', fft2, H)
            ifft2 = np.fft.ifftshift(np.fft.ifft2(filtered, axes=(0,1)), axes=(0,1))

            self.h_xyzJ_single_to_xye_det[...,i] = ifft2
        
    # Generates detector irradiances (after microlenses) due to a single dipole
    # Input: R3S2toR.xyzj_list with a single entry
    # Output: R2toR.xy object
    def xyzj_single_to_xy_det(self, dip):
        edet = self.xyzj_single_to_xye_det(dip)
        return self.fourf.xye_to_xy_det(edet)

    # Generates the irradiance pattern due to a single dipole distribution
    # Input: R3S2toR.xyzJ_list with a single entry
    # Output: R2toR.xy object
    def xyzJ_single_to_xy_det(self, dist):
        self.precompute_xyzJ_single_to_xy_det(dist.data_xyz[0])
        out = np.einsum('ijk,k->ij', self.h_xyzJ_single_to_xy_det, dist.data_J[0])

        return R2toR.xy(out, 
                        title='Lightfield detector irradiance',
                        fov=[-self.fourf.fov/2, self.fourf.fov/2],
                        plotfov=[-self.fourf.plotfov/2, self.fourf.plotfov/2])

    def precompute_xyzJ_single_to_xy_det(self, dist):
        self.precompute_xyzJ_single_to_xye_det(dist)
        self.h_xyzJ_single_to_xy_det = np.zeros((self.fourf.npx[0], self.fourf.npx[1], 6), dtype='float32')

        # Compute gaunt coeffs
        G = utilsh.gaunt_l1l1_tol0l2()

        # Compute matrix
        out = np.real(np.einsum('ijkl,ijkm,nlm->ijn',
                                self.h_xyzJ_single_to_xye_det,
                                self.h_xyzJ_single_to_xye_det.conj(),
                                G)).astype('float32')

        # Downsample and store
        self.h_xyzJ_single_to_xy_det = out.reshape(self.fourf.npx[0], self.fourf.ss, self.fourf.npx[1], self.fourf.ss, 6).sum(axis=(1,3))

    # Generates the irradiance pattern due to several dipole distributions
    # This is a slow path for dense objects.
    # Input: R3S2toR.xyzJ_list
    # Output: R2toR.xy object
    def xyzJ_list_to_xy_det(self, dist):
        out = np.zeros(self.fourf.npx)
        for m in range(dist.M):
            distm = R3S2toR.xyzJ_list([dist.data_xyz[m]], [dist.data_J[m]])
            out += self.xyzJ_single_to_xy_det(distm).data
        return R2toR.xy(out, 
                        title=self.fourf.irrad_title,
                        fov=[-self.fourf.fov/2, self.fourf.fov/2],
                        plotfov=[-self.fourf.plotfov/2, self.fourf.plotfov/2])

    # Generates the irradiance pattern on a lfcamera from a dense xyzJ array.
    # Alternate name for xyzJ_to_xy_det    
    # This is a faster path than xyzJ_list_to_xy_det
    # Input: R3S2toR.xyzJ
    # Output: R2toR.xy object
    def fwd(self, xyzJ):
        log.info('Applying forward operator')
        # Reshape xyzJ to uvstzJ
        uu, vv, ss, tt, zz, jj, vp, tp = self.H_UvStzJ_to_UvSt_det.shape
        uvstzJ_shape = (uu, vp, uu, tp, zz, jj)
        uvstzJ = xyzJ.data.reshape(uvstzJ_shape)

        # FFT uvstzJ to UvStzJ (with Fourier shifting)
        uvstzJ_shift = np.fft.ifftshift(uvstzJ, axes=(0,2))
        UvStzJ_shift = np.fft.rfftn(uvstzJ_shift, axes=(0,2))

        # Forward model matrix multiplication
        # Sum over mnop
        UvSt_shift = np.einsum('ijklmnop,iokpmn->ijkl', self.H_UvStzJ_to_UvSt_det, UvStzJ_shift)

        # FFT UvSt to uvst (with Fourier deshifting)
        uvst_shift = np.fft.irfftn(UvSt_shift, s=(uu,uu), axes=(0,2))
        uvst = np.fft.fftshift(uvst_shift, axes=(0,2))

        # Reshape uvst to xy
        xy = uvst.reshape(self.fourf.npx[0], self.fourf.npx[0])

        return R2toR.xy(xy,
                        px_dims=2*(xyzJ.vox_dims[0]*vp/vv,),
                        title=self.fourf.irrad_title,
                        fov=[-self.fourf.fov/2, self.fourf.fov/2],
                        plotfov=[-self.fourf.fov/2, self.fourf.fov/2])

    def precompute_fwd(self, input_shape, input_vox_dims):
        # Precompute UvStzJ_to_UvSt_det
        uu, vv, ss, tt, zz, jj = input_shape
        matrix_shape = (uu, self.ulenpx, int(np.floor(ss/2 + 1)), self.ulenpx, zz, jj, vv, tt)
        self.H_UvStzJ_to_UvSt_det = np.zeros(matrix_shape, dtype=np.complex64)

        log.info('Computing psfs')
        for i in trange(input_shape[1], desc='Outer loop'):
            for j in trange(input_shape[3], desc='Inner loop', leave=False):
                for k in range(input_shape[4]):
                    x = (i + 0.5 - 0.5*input_shape[1])*input_vox_dims[0] # i2x
                    y = (j + 0.5 - 0.5*input_shape[3])*input_vox_dims[1] # j2y
                    z = (k + 0.5 - 0.5*input_shape[4])*input_vox_dims[2] # k2z
                    self.precompute_xyzJ_single_to_xy_det([x,y,z])
                    
                    h_uvstJ = self.h_xyzJ_single_to_xy_det.reshape((uu,self.ulenpx,ss,self.ulenpx,jj))
                    h_shift = np.fft.ifftshift(h_uvstJ, axes=(0,2))
                    entry = np.fft.rfft2(h_shift, axes=(0,2))

                    self.H_UvStzJ_to_UvSt_det[:,:,:,:,k,:6,i,j] = entry

    # Generates the pseudoinverse solution
    # Requires self.Hp and self.H_UvStzJ_to_UvSt_det to have been precomputed
    # Input: R3S2toR.xy
    # Output: R2toR.xyzJ object
    def pinv(self, xy, out_vox_dims=[.1,.1,.1]):
        log.info('Applying pseudoinverse operator')
        U, v, S, t, z, J, vp, tp = self.H_UvStzJ_to_UvSt_det.shape
        
        # Resort and FT data
        uvst = xy.data.reshape((U,v,U,t)).astype('float32')
        uvst_shift = np.fft.ifftshift(uvst, axes=(0,2))
        UvSt_shift = np.fft.rfftn(uvst_shift, axes=(0,2))

        # Apply pinv operator
        UvStzJ_shift = np.einsum('ikjlopqr,ijkl->iokpqr', self.Hp, UvSt_shift)
        # UvStzJ_shift = np.einsum('ikmnopqr,ikmn,ikjlmn,ijkl->iokpqr', VVall, 1/SSall, UUall, UvSt_shift)

        # FFT UvSt to uvst (with Fourier deshifting)
        uvstzJ_shift = np.fft.irfftn(UvStzJ_shift, s=(U,U), axes=(0,2))
        uvstzJ = np.fft.fftshift(uvstzJ_shift, axes=(0,2))

        # Reshape uvst to xy
        xyzJ = uvstzJ.reshape((U*vp, U*vp, z, J))
        # xyzJ = np.flip(xyzJ, axis=2) # Kludge for now. I think uvst is accidentally transposed.
        
        return R3S2toR.xyzJ(xyzJ, vox_dims=out_vox_dims, title='Reconstructed object')

    # Precompute the pseudoinverse matrix
    # Requires self.H_UvStzJ_to_UvSt_det to have been precomputed
    def precompute_pinv(self, eta=0):
        log.info('Computing SVD')        
        U, v, S, t, z, J, vp, tp = self.H_UvStzJ_to_UvSt_det.shape
        sort = np.moveaxis(self.H_UvStzJ_to_UvSt_det, [0,2,1,3,6,7,4,5], [0,1,2,3,4,5,6,7])
        InvOI = sort.reshape((U*S, v*t, vp*tp*z*J))
        Inv, O, I = InvOI.shape
        K = np.min([I,O])
        UU = np.zeros((Inv, K, O), dtype=np.complex64)
        SS = np.zeros((Inv, K), dtype=np.float32)
        VV = np.zeros((Inv, K, I), dtype=np.complex64)
        for i in tqdm(range(Inv)):
            uu, ss, vv = np.linalg.svd(InvOI[i,:,:], full_matrices=False)
            UU[i,:] = uu            
            SS[i,:] = ss
            VV[i,:] = vv

        UUall = UU.reshape((U,S,v,t,v,t))
        SSall = SS.reshape((U,S,v,t))
        VVall = VV.reshape((U,S,v,t,vp,tp,z,J))

        # Reconstruct
        log.info('Computing pseduoinverse operator')
        SSreg = np.where(SSall > 1e-7, SSall/(SSall**2 + eta), 0) # Regularize
        self.Hp = np.einsum('ikmnopqr,ikmn,ikjlmn->ikjlopqr', VVall, SSreg, UUall)
