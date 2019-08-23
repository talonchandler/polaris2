import numpy as np
import vtk
from vtk.util import numpy_support as ns
from polaris2.geomvis import utilmpl, utilvtk

# Sparse mapping from R3 to R3
# data_xyz is an Mx3 array of input positions
# data_ijk is an Mx3 array of output positions
class xyz_list:
    def __init__(self, data_xyz, data_ijk, shape=[10,10,4], vmin=0, vmax=None, xlabel='', title='', invert=True):
        self.data_xyz = data_xyz
        self.data_ijk = data_ijk

        self.shape = shape

        self.xlabel = xlabel
        self.title = title

        # Setup renderer
        self.ren, self.renWin, self.iren = utilvtk.setup_render()

    def build_actors(self):
        M = self.data_xyz.shape[0]
        
        # Calculate points
        r = 0.1
        starts = self.data_xyz - r*self.data_ijk
        ends = self.data_xyz + r*self.data_ijk

        # Interleave starts and ends
        points_array = np.empty((2*M, 3))
        points_array[0::2] = starts
        points_array[1::2] = ends

        # Calculate line connections
        lines_array = []
        for i in range(M):
            lines_array += [2, 2*i, 2*i+1] # length, index0, index1
        lines_array = np.array(lines_array)

        # Set Points to vtk array format
        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(ns.numpy_to_vtk(points_array, deep=True))

        # Set Lines to vtk array format
        vtk_lines = vtk.vtkCellArray()
        vtk_lines.GetData().DeepCopy(ns.numpy_to_vtk(lines_array))
        vtk_lines.SetNumberOfCells(M)

        # Set poly_data
        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(vtk_points)
        poly_data.SetLines(vtk_lines)

        # Set tube radius
        tube_filter = vtk.vtkTubeFilter()
        tube_filter.SetInputData(poly_data)
        tube_filter.SetNumberOfSides(50)
        tube_filter.SetRadius(0.03)
        # tube_filter.SetVaryRadiusToVaryRadiusByScalar() # Possible TODO
        tube_filter.CappingOn()
        tube_filter.Update()

        # Original
        poly_mapper = vtk.vtkPolyDataMapper()
        poly_mapper.SetInputConnection(tube_filter.GetOutputPort())
        poly_mapper.ScalarVisibilityOn()
        poly_mapper.SetScalarModeToUsePointFieldData()
        poly_mapper.SelectColorArray("Colors")
        poly_mapper.Update()

        # Color
        cols = np.zeros_like(points_array)
        cols[0::2] = 255*(self.data_ijk*0.5 + 0.5)
        cols[1::2] = 255*(self.data_ijk*0.5 + 0.5)
        vtk_colors = ns.numpy_to_vtk(cols, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        vtk_colors.SetName("Colors")
        poly_data.GetPointData().SetScalars(vtk_colors)

        # Set actor
        actor = vtk.vtkActor()
        actor.SetMapper(poly_mapper)
        actor.GetProperty().SetLineWidth(2)
        self.ren.AddActor(actor)

        # Draw extras
        utilvtk.draw_outer_box(self.ren, *self.shape)
        utilvtk.draw_axes(self.ren, *self.shape)
        
        # Set cameras
        dist = 1.1*np.linalg.norm(self.shape)
        self.ren.GetActiveCamera().SetPosition(np.array([1,-1,1])*dist)
        self.ren.GetActiveCamera().SetViewUp([0,0,1])
        self.ren.ResetCamera()

    def increment_camera(self, az):
        self.ren.GetActiveCamera().Azimuth(az)
        
    def plot(self, f, fc, ss):
        ax = utilmpl.plot_template(f, fc, xlabel=self.xlabel, title=self.title,
                                   scale_bar=False, bump=1.2)

        utilvtk.vtk2imshow(self.renWin, ax[0], ss)
        ax[0].axis('off')
        ax[1].axis('off')
