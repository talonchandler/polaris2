import vtk
from vtk.util import numpy_support
import numpy as np
from polaris2.geomvis import utilmpl, utilvtk, utilsh

# List of single dipoles in [[x0,y0,z0,sx0,sy0,sz0], [x0,y0,z0,sx0,sy0,sz0]] form.
class xyzj_list:
    def __init__(self, data, shape=[10,10,2.5], xlabel='', title=''):
        self.data = data
        self.shape = shape # um

        self.xlabel = xlabel
        self.title = title

        # Setup renderer
        self.ren, self.renWin, self.iren = utilvtk.setup_render()

    def build_actors(self):
        # Add double arrows for dipoles
        for dipole in self.data:
            utilvtk.draw_double_arrow(self.ren, *dipole) 
        utilvtk.draw_origin_dot(self.ren)
        utilvtk.draw_outer_box(self.ren, *self.shape)
        utilvtk.draw_axes(self.ren, *self.shape)
        
        # Set cameras
        self.ren.GetActiveCamera().SetPosition([1,-1,1])
        self.ren.GetActiveCamera().SetViewUp([0,0,1])
        self.ren.ResetCamera()

    def increment_camera(self, az):
        self.ren.GetActiveCamera().Azimuth(az)

    def plot(self, f, fc):
        ax = utilmpl.plot_template(f, fc, xlabel=self.xlabel, title=self.title,
                                scale_bar=False, bump=1.2)

        utilvtk.vtk2imshow(self.renWin, ax[0])
        ax[0].axis('off')
        ax[1].axis('off')
