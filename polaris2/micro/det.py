import numpy as np
from polaris2.geomvis import R2toR, R2toC2, R3S2toR, util

# Simulates a linear dipole imaged by 4f detection system.
class FourF:

    def __init__(self, NA=1.4, M=63, n0=1.5, lamb=0.546, wpx_real=7.4,
                 npx=(7*17, 7*17), ss=3, plotfov=10, irrad_title='4$f$ detector irradiance'):
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

    # Generates the irradiance pattern due to a single dipole
    # This is an abstraction over the main functions in this class
    # Input: R3S2toR.xyzj_list with a single entry
    # Output: R2toC2.xy object
    def dip_to_det(self, dip):
        ebfp = self.dip_to_ebfp(dip)
        eim = self.ebfp_to_eim(ebfp)
        return self.e_to_i(eim)
        
    # Generates the electric field pattern in the bfp due to a single dipole
    # Input: R3S2toR.xyzj_list with a single entry
    # Output: R2toC2.xy object
    # Based on: Backer, A. S., & Moerner, W. E. (2014)
    # http://dx.doi.org/10.1021/jp501778z
    def dip_to_ebfp(self, dip):
        rx, ry, rz, sx, sy, sz = dip.data[0]

        # Coordinates
        if self.npxss % 2 == 0:
            xmin = -0.5*self.npxss*self.wbfp
            xmax = (0.5*self.npxss-1)*self.wbfp
        else:
            xmin = -0.5*(self.npxss-1)*self.wbfp
            xmax = 0.5*(self.npxss-1)*self.wbfp
        x = np.linspace(xmin, xmax, self.npxss)
        y = np.linspace(xmin, xmax, self.npxss)
        
        taux, tauy = np.meshgrid(x, y)
        tauphi = np.arctan2(tauy, taux)
        abstau = np.sqrt(taux**2 + tauy**2)
        rho = abstau/self.num

        # Apodization and phase calculations
        apod = np.where(abstau < self.nuc/2, 1, 0)
        sine_apod = (1 - apod*rho**2)**(-0.25)
        sqrtrho = np.sqrt(1 - apod*rho**2)
        tphase = np.exp(1j*2*np.pi*(taux*rx + tauy*ry))
        aphase = np.exp(1j*2*np.pi*rz*np.sqrt(self.num**2 - apod*abstau**2))

        # Polarization calculations
        gxx = (np.sin(tauphi))**2 + (np.cos(tauphi))**2*sqrtrho
        gxy = 0.5*np.sin(2*tauphi)*(sqrtrho - 1)
        gxz = rho*np.cos(tauphi)

        # gyx == gxy
        gyy = (np.cos(tauphi))**2 + (np.sin(tauphi))**2*sqrtrho
        gyz = rho*np.sin(tauphi)

        pre = apod*sine_apod*tphase*aphase
        ebfpx = pre*(gxx*sx + gxy*sy + gxz*sz)
        ebfpy = pre*(gxy*sx + gyy*sy + gyz*sz)

        # Monopole 
        if sx == 0 and sy == 0 and sz == 0:
            ebfpx = pre
            ebfpy = pre

        # Return BFP
        return R2toC2.xy(np.stack([ebfpx, ebfpy], axis=-1),
                         circle=True, title='Scaled back focal plane fields',
                         xlabel='$\\textrm{2NA}/\lambda$',
                         toplabel='$E_x$', bottomlabel='$E_y$',
                         fov=[xmin, xmax], plotfov=[-self.nuc/2, self.nuc/2],
                         colormax=1.5)

    # Propagates the electric fields from the bfp to the image plane 
    # Input: R2toC2.xy object
    # Output: R2toC2.xy object
    def ebfp_to_eim(self, ebfp):
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
    
    # Generates irradiance from electric fields
    # Input: R2toC2.xy object
    # Output: R2toR.xy object
    def e_to_i(self, e):
        # Calculate irradiance
        irr = np.sum(np.abs(e.data)**2, axis=-1) 

        # "Undo" supersampling by summing over squares
        irrpx = irr.reshape(self.npx[0], self.ss, self.npx[1], self.ss).sum(axis=(1,3))

        # Return 
        return R2toR.xy(irrpx, 
                        title=self.irrad_title,
                        fov=[-self.fov/2, self.fov/2],
                        plotfov=[-self.plotfov/2, self.plotfov/2],
                        xlabel=str(self.plotfov)+' $\mu$m')

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

    # Generates detector fields (after microlenses) due to a single dipole
    # Input: R3S2toR.xyzj_list with a single entry
    # Output: R2toR.xy object
    def dip_to_det(self, dip):

        # Use FourF to calculate electric field in nominal image plane
        ebfp = self.fourf.dip_to_ebfp(dip)
        eim = self.fourf.ebfp_to_eim(ebfp)

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

        edet = R2toC2.xy(ifft2,
                         fov=[-self.fourf.fov/2, self.fourf.fov/2],
                         plotfov=[-self.fourf.plotfov/2, self.fourf.plotfov/2],
                         title='Lightfield detector fields',
                         xlabel=str(self.fourf.plotfov)+' $\mu$m',
                         toplabel='$E_x$', bottomlabel='$E_y$')

        return self.fourf.e_to_i(edet)
