# VTK utilities
import vtk
from vtk.util import numpy_support

import numpy as np
from scipy.spatial import ConvexHull
from polaris2.geomvis import utilsh

# Setup vtk render and renderwindow
def setup_render(size=2000):
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetSize(size, size)
    ren.SetBackground([255,255,255])
    renWin.OffScreenRenderingOff()
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    return ren, renWin, iren

# Plots a vtk renderWindow and in a matplotlib axis with imshow
def vtk2imshow(renWin, ax):
    # Render
    renWin.OffScreenRenderingOn()
    renWin.Render()

    # Filter renWin
    image_filter = vtk.vtkWindowToImageFilter()
    image_filter.SetInput(renWin)
    image_filter.SetScale(1)
    image_filter.SetInputBufferTypeToRGB()
    image_filter.ReadFrontBufferOff()
    image_filter.Update()

    # Convert to numpy array 
    im = image_filter.GetOutput()
    rows, cols, _ = im.GetDimensions()
    sc = im.GetPointData().GetScalars()
    a = numpy_support.vtk_to_numpy(sc)
    a = a.reshape(rows, cols, -1)
    ax.imshow(a, origin='lower')

# Make axes acotr for vtk plotting
def make_axes():
    axes = vtk.vtkAxesActor()
    axes.SetShaftTypeToCylinder()
    axes.SetXAxisLabelText('')
    axes.SetYAxisLabelText('')
    axes.SetZAxisLabelText('')
    axes.SetTotalLength(1.5, 1.5, 1.5)
    axes.SetCylinderRadius(0.75 * axes.GetCylinderRadius())
    axes.SetConeRadius(1.5 * axes.GetConeRadius())
    axes.SetSphereRadius(1.5 * axes.GetSphereRadius())
    return axes

def draw_unlit_line(ren, x1, y1, z1, x2, y2, z2, color=[0,0,0], width=10):
    line = vtk.vtkLineSource()
    line.SetPoint1(x1,y1,z1)
    line.SetPoint2(x2,y2,z2)

    linem = vtk.vtkPolyDataMapper()
    linem.SetInputConnection(line.GetOutputPort())
    linea = vtk.vtkActor()
    linea.SetMapper(linem)
    linea.GetProperty().SetLineWidth(width)
    linea.GetProperty().SetColor(color)
    linea.GetProperty().SetLighting(0)
    ren.AddActor(linea)

def draw_axes(ren, x, y, z):
    dx = x/10
    zz = np.min([x, y, z])/100
    draw_unlit_line(ren,-x/2+zz,-y/2+zz,-z/2+zz, -x/2+dx+zz,-y/2+zz,-z/2+zz, color=[1,0,0], width=15)
    draw_unlit_line(ren,-x/2+zz,-y/2+zz,-z/2+zz, -x/2+zz,-y/2+dx+zz,-z/2+zz, color=[0,1,0], width=15)
    draw_unlit_line(ren,-x/2+zz,-y/2+zz,-z/2+zz, -x/2+zz,-y/2+zz,-z/2+dx+zz, color=[0,0,1], width=15)

def draw_outer_box(ren, x, y, z):
    # Botton layer
    draw_unlit_line(ren,-x/2,-y/2,-z/2, +x/2,-y/2,-z/2)
    draw_unlit_line(ren,+x/2,-y/2,-z/2, +x/2,+y/2,-z/2)
    draw_unlit_line(ren,+x/2,+y/2,-z/2, -x/2,+y/2,-z/2)
    draw_unlit_line(ren,-x/2,+y/2,-z/2, -x/2,-y/2,-z/2)

    # Top layer
    draw_unlit_line(ren,-x/2,-y/2,+z/2, +x/2,-y/2,+z/2)
    draw_unlit_line(ren,+x/2,-y/2,+z/2, +x/2,+y/2,+z/2)
    draw_unlit_line(ren,+x/2,+y/2,+z/2, -x/2,+y/2,+z/2)
    draw_unlit_line(ren,-x/2,+y/2,+z/2, -x/2,-y/2,+z/2)

    # Sides
    draw_unlit_line(ren,-x/2,-y/2,-z/2, -x/2,-y/2,+z/2)
    draw_unlit_line(ren,+x/2,-y/2,-z/2, +x/2,-y/2,+z/2)
    draw_unlit_line(ren,-x/2,+y/2,-z/2, -x/2,+y/2,+z/2)
    draw_unlit_line(ren,+x/2,+y/2,-z/2, +x/2,+y/2,+z/2)

def draw_origin_dot(ren):
    dot = vtk.vtkSphereSource()
    dot.SetRadius(.1)
    dotm = vtk.vtkPolyDataMapper()
    dotm.SetInputConnection(dot.GetOutputPort())
    dota = vtk.vtkActor()
    dota.SetMapper(dotm)
    dota.GetProperty().SetColor([0,0,0])
    ren.AddActor(dota)

def draw_double_arrow(ren, x, y, z, sx, sy, sz):
    for i in range(2):
        arrow = vtk.vtkArrowSource()
        arrow.SetTipResolution(50)
        arrow.SetShaftResolution(50)

        arrowm = vtk.vtkPolyDataMapper()
        arrowm.SetInputConnection(arrow.GetOutputPort())
        arrowa = vtk.vtkActor()
        arrowa.SetMapper(arrowm)
        arrowa.GetProperty().SetColor([.5,.5,.5])
        arrowa.SetScale(1)

        tp = utilsh.xyz2tp(sx, sy, sz)
        arrowa.RotateWXYZ(-90, 0, 1, 0) # Align with Z axis
        arrowa.RotateWXYZ(np.rad2deg(tp[0]), 0, 1, 0)
        arrowa.RotateWXYZ(np.rad2deg(tp[1]), 0, 0, 1)
        arrowa.SetPosition(x, y, z)
        if i == 1:
            arrowa.RotateX(180)
            arrowa.RotateZ(180)

        ren.AddActor(arrowa)

def draw_sphere_function(ren, xyz, pradii, nradii):
    # Plot each lobe
    for i, radii in enumerate([pradii, nradii]):
        all_xyz = np.einsum('i,ij->ij', radii, xyz)

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
        cols = 255*np.ones((xyz.shape[0], 3))
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
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # Color the actor
        colors = vtk.vtkNamedColors()

        ren.AddActor(actor)