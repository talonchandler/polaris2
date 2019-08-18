import vtk
import numpy as np
from polaris2.geomvis import utilmpl, utilvtk, utilsh
import logging
log = logging.getLogger('log')

class Jeven:
    def __init__(self, data, xlabel='', title='', N=2**14, lmax=None):
                 
        self.data = np.array(data)
        self.N = N # points on the sphere

        # Plotting
        self.xlabel = xlabel
        self.title = title
        
        # Setup renderer
        self.ren, self.renWin, self.iren = utilvtk.setup_render()

        # Calculate dimensions
        if lmax is None:
            self.lmax, mm = utilsh.j2lm(len(self.data) - 1)
        else:
            self.lmax = lmax
        self.J = utilsh.maxl2maxj(self.lmax)

        # Fill the rest of the last l band with zeros
        if self.data.shape[-1] != self.J:
            temp = np.zeros(self.J)
            temp[:self.data.shape[-1]] = self.data
            self.data = temp
        else:
            self.data = data

        # Calc points for spherical plotting
        self.xyz = utilsh.fibonacci_sphere(N, xyz=True)
        self.B = utilsh.calcB(self.N, self.J)
        self.Binv = np.linalg.pinv(self.B, rcond=1e-15)

    def __mul__(self, other):
        result = np.einsum('ijk,j,k->i', self.G, self.data, other.data)
        return Jeven(result)

    def __rmul__(self, other):
        return self.__mul__(other)

    def precompute_tripling(self):
        Jout = utilsh.maxl2maxj(2*self.lmax)
        self.G = utilsh.G_real_mult_tensor(Jout, self.J)
    
    def build_actors(self):
        # Calculate positive and negative lobes
        radii = np.einsum('ij,j->i', self.B, self.data[-1])
        pradii = radii.clip(min=0)
        nradii = -radii.clip(max=0)

        utilvtk.draw_sphere_function(self.ren, self.xyz, pradii, nradii)

        # Make orientation axes
        axes1 = utilvtk.make_axes()
        self.om1 = vtk.vtkOrientationMarkerWidget()
        self.om1.SetOrientationMarker(axes1)
        self.om1.SetViewport(0, -0.125, 0.375, 0.25)
        self.om1.SetInteractor(self.iren)
        self.om1.EnabledOn()

        # Set cameras
        self.ren.GetActiveCamera().SetPosition([1,1,1])
        self.ren.GetActiveCamera().SetViewUp([0,0,1])
        self.ren.ResetCamera()

    def increment_camera(self, az):
        self.ren.GetActiveCamera().Azimuth(az)
        
    def plot(self, f, fc):
        ax = utilmpl.plot_template(f, fc, xlabel=self.xlabel, title=self.title,
                                scale_bar=False)

        # Plot to axis
        utilvtk.vtk2imshow(self.renWin, ax[0])
        
        # Turn off axis outline
        ax[0].axis('off')
        
        # Colorbar        
        vmin = -1
        vmax = 1
        x = np.linspace(vmin, vmax, 100)
        xx, yy = np.meshgrid(x, x)

        ax[1].imshow(yy, vmin=vmin, vmax=vmax, cmap='bwr',
                     extent=[0,1,vmin,vmax], aspect='auto',
                     interpolation='bicubic', origin='lower')
        ax[1].annotate('${:.2g}$'.format(np.max(vmax)), xy=(0,0), xytext=(1.8, 1.0), textcoords='axes fraction', va='center', ha='left')
        ax[1].annotate('$-{:.2g}$'.format(np.max(vmax)), xy=(0,0), xytext=(1.8, 0), textcoords='axes fraction', va='center', ha='left')        
        ax[1].yaxis.set_ticks([vmin, vmax])
        ax[1].set_yticklabels(['', ''])

# Interactivity code
# For possible resurrection later
#     def interact(self):
#         self.build_actors()

#         # Slider is for testing (delete later)
#         # Setup a slider widget
#         tubeWidth = 0.008
#         sliderLength = 0.008
#         titleHeight = 0.04
#         labelHeight = 0.04

#         sliderRep1 = vtk.vtkSliderRepresentation2D()

#         sliderRep1.SetMinimumValue(0.0)
#         sliderRep1.SetMaximumValue(1.0)
#         sliderRep1.SetValue(1.0)
#         sliderRep1.SetTitleText("Y Height")

#         sliderRep1.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
#         sliderRep1.GetPoint1Coordinate().SetValue(.8, .1)
#         sliderRep1.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
#         sliderRep1.GetPoint2Coordinate().SetValue(.95, .1)

#         sliderRep1.SetTubeWidth(tubeWidth)
#         sliderRep1.SetSliderLength(sliderLength)
#         sliderRep1.SetTitleHeight(titleHeight)
#         sliderRep1.SetLabelHeight(labelHeight)
        
#         sliderRep1.GetTubeProperty().SetColor([0,0,0])
#         sliderRep1.GetLabelProperty().SetColor([0,0,0])
#         sliderRep1.GetSliderProperty().SetColor([0,0,0])
#         sliderRep1.GetTitleProperty().SetColor([0,0,0])        

#         sliderWidget1 = vtk.vtkSliderWidget()
#         sliderWidget1.SetInteractor(self.iren)
#         sliderWidget1.SetRepresentation(sliderRep1)
#         sliderWidget1.SetAnimationModeToAnimate()
#         sliderWidget1.EnabledOn()

#         sliderWidget1.AddObserver(vtk.vtkCommand.InteractionEvent, SliderCallback1(self.actor))

#         # Keep this for interactive
#         self.iren.Initialize()
#         self.iren.Start()
        
# class SliderCallback1():
#     def __init__(self, obj):
#         self.obj = obj

#     def __call__(self, caller, ev):
#         sliderWidget = caller
#         value = sliderWidget.GetRepresentation().GetValue()
#         if value > 0.5:
#             self.obj.VisibilityOn()
#         else:
#             self.obj.VisibilityOff()
