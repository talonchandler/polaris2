import vtk
from vtk.util import numpy_support
from scipy.spatial import ConvexHull
import numpy as np
from polaris2.geomvis import util, gaunt
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
        self.ren, self.renWin, self.iren = util.setup_render()

        # Calculate dimensions
        if lmax is None:
            self.lmax, mm = util.j2lm(len(self.data) - 1)
        else:
            self.lmax = lmax
        self.J = util.maxl2maxj(self.lmax)

        # Fill the rest of the last l band with zeros
        if self.data.shape[-1] != self.J:
            temp = np.zeros(self.J)
            temp[:self.data.shape[-1]] = self.data
            self.data = temp
        else:
            self.data = data

        # Calculate odf to sh matrix (B) and inverse
        self.xyz = util.fibonacci_sphere(self.N, xyz=True)
        tp = util.fibonacci_sphere(self.N)
        B = np.zeros((self.N, self.J))
        for (n, j), x in np.ndenumerate(B):
            l, m = util.j2lm(j)
            B[n, j] = util.spZnm(l, m, tp[n,0], tp[n,1])
        self.B = B
        self.Binv = np.linalg.pinv(self.B, rcond=1e-15)

    def __mul__(self, other):
        result = np.einsum('ijk,j,k->i', self.G, self.data, other.data)
        return Jeven(result)

    def __rmul__(self, other):
        return self.__mul__(other)

    def precompute_tripling(self):
        Jout = util.maxl2maxj(2*self.lmax)
        self.G = np.zeros((Jout, self.J, self.J))
        for j in range(Jout):
            for jp in range(self.J):
                for jpp in range(self.J):
                    l, m = util.j2lm(j)
                    lp, mp = util.j2lm(jp)
                    lpp, mpp = util.j2lm(jpp)
                    self.G[j,jp,jpp] = gaunt.gauntR(l,lp,lpp,m,mp,mpp)

    def build_actors(self):
        # Calculate positive and negative lobes
        radii = np.einsum('ij,i->j', self.Binv, self.data)
        pradii = radii.clip(min=0)
        nradii = -radii.clip(max=0)

        # Plot each lobe
        for i, radii in enumerate([pradii, nradii]):
            all_xyz = np.einsum('i,ij->ij', radii, self.xyz)

            xyz = util.fibonacci_sphere(self.N, xyz=True)
            ch = ConvexHull(xyz)
            all_faces = []
            all_faces.append(ch.simplices)

            all_xyz = np.ascontiguousarray(all_xyz)
            all_xyz_vtk = numpy_support.numpy_to_vtk(all_xyz, deep=True)

            all_faces = np.concatenate(all_faces)
            all_faces = np.hstack((3 * np.ones((len(all_faces), 1)), all_faces))

            ncells = len(all_faces)
            all_faces = np.ascontiguousarray(all_faces.ravel(), dtype='i8')
            all_faces_vtk = numpy_support.numpy_to_vtkIdTypeArray(all_faces, deep=True)

            points = vtk.vtkPoints()
            points.SetData(all_xyz_vtk)

            cells = vtk.vtkCellArray()
            cells.SetCells(ncells, all_faces_vtk)

            polydata = vtk.vtkPolyData()
            polydata.SetPoints(points)
            polydata.SetPolys(cells)

            # TODO: Generalize colormaps
            cols = 255*np.ones((self.N, 3))
            if i == 0:
                cols[:,1] = 255*(1-radii/np.max(radii))
                cols[:,2] = 255*(1-radii/np.max(radii))
            else:
                cols[:,0] = 255*(1-radii/np.max(radii))
                cols[:,1] = 255*(1-radii/np.max(radii))
            vtk_colors = numpy_support.numpy_to_vtk(cols, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
            polydata.GetPointData().SetScalars(vtk_colors)
            
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(polydata)

            # Create actor
            self.actor = vtk.vtkActor()
            self.actor.SetMapper(mapper)

            # Color the actor
            colors = vtk.vtkNamedColors()

            self.ren.AddActor(self.actor)

        # Make orientation axes
        axes1 = util.make_axes()
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
        ax = util.plot_template(f, fc, xlabel=self.xlabel, title=self.title,
                                scale_bar=False)

        # Plot to axis
        util.vtk2imshow(self.renWin, ax[0])
        
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
        ax[1].annotate('{:.2g}'.format(np.max(vmax)), xy=(0,0), xytext=(0, 1.05), textcoords='axes fraction', va='center', ha='left')
        ax[1].annotate('0', xy=(0,0), xytext=(1.8, 0), textcoords='axes fraction', va='center', ha='left')        
        ax[1].yaxis.set_ticks([vmin, vmax])
        ax[1].set_yticklabels(['', ''])

    def interact(self):
        self.build_actors()

        # Slider is for testing (delete later)
        # Setup a slider widget
        tubeWidth = 0.008
        sliderLength = 0.008
        titleHeight = 0.04
        labelHeight = 0.04

        sliderRep1 = vtk.vtkSliderRepresentation2D()

        sliderRep1.SetMinimumValue(0.0)
        sliderRep1.SetMaximumValue(1.0)
        sliderRep1.SetValue(1.0)
        sliderRep1.SetTitleText("Y Height")

        sliderRep1.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
        sliderRep1.GetPoint1Coordinate().SetValue(.8, .1)
        sliderRep1.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
        sliderRep1.GetPoint2Coordinate().SetValue(.95, .1)

        sliderRep1.SetTubeWidth(tubeWidth)
        sliderRep1.SetSliderLength(sliderLength)
        sliderRep1.SetTitleHeight(titleHeight)
        sliderRep1.SetLabelHeight(labelHeight)
        
        sliderRep1.GetTubeProperty().SetColor([0,0,0])
        sliderRep1.GetLabelProperty().SetColor([0,0,0])
        sliderRep1.GetSliderProperty().SetColor([0,0,0])
        sliderRep1.GetTitleProperty().SetColor([0,0,0])        

        sliderWidget1 = vtk.vtkSliderWidget()
        sliderWidget1.SetInteractor(self.iren)
        sliderWidget1.SetRepresentation(sliderRep1)
        sliderWidget1.SetAnimationModeToAnimate()
        sliderWidget1.EnabledOn()

        sliderWidget1.AddObserver(vtk.vtkCommand.InteractionEvent, SliderCallback1(self.actor))

        # Keep this for interactive
        self.iren.Initialize()
        self.iren.Start()
        
class SliderCallback1():
    def __init__(self, obj):
        self.obj = obj

    def __call__(self, caller, ev):
        sliderWidget = caller
        value = sliderWidget.GetRepresentation().GetValue()
        if value > 0.5:
            self.obj.VisibilityOn()
        else:
            self.obj.VisibilityOff()
