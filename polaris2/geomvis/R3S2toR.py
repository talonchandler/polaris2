import vtk
from vtk.util import numpy_support
import numpy as np
from polaris2.geomvis import utilv, util

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

        # Setup renderer
        self.ren, self.renWin, self.iren = utilv.setup_render()

    def build_actors(self):
        # Add double arrows for dipoles
        for dipole in self.data:
            utilv.draw_double_arrow(self.ren, *dipole) 
        utilv.draw_origin_dot(self.ren)
        utilv.draw_outer_box(self.ren, *self.shape)
        utilv.draw_axes(self.ren, *self.shape)
        
        # Set cameras
        self.ren.GetActiveCamera().SetPosition([1,-1,1])
        self.ren.GetActiveCamera().SetViewUp([0,0,1])
        self.ren.ResetCamera()

    def increment_camera(self, az):
        self.ren.GetActiveCamera().Azimuth(az)

    def plot(self, f, fc):
        ax = util.plot_template(f, fc, xlabel=self.xlabel, title=self.title,
                                scale_bar=False, bump=1.2)

        # Plot to axis
        utilv.vtk2imshow(self.renWin, ax[0])
        
        # Turn off axis outline
        ax[0].axis('off')
        
        # Colorbar off
        ax[1].axis('off')
