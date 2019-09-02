# VTK utilities
import vtk
from vtk.util import numpy_support

import numpy as np
from scipy.spatial import ConvexHull
from polaris2.geomvis import utilsh

# Setup vtk render and renderwindow
def setup_render():
    ren = vtk.vtkRenderer()
    ren.UseFXAAOn() # anti-aliasing on
    
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    ren.SetBackground([255,255,255])
    renWin.OffScreenRenderingOff()
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    return ren, renWin, iren

# Plots a vtk renderWindow and in a matplotlib axis with imshow
def vtk2imshow(renWin, ax, ss=1):
    # Render
    renWin.OffScreenRenderingOn()
    renWin.SetSize(int(1000*ss), int(1000*ss))
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

# Make axes actor for vtk plotting
def make_axes():
    axes = vtk.vtkAxesActor()
    axes.SetShaftTypeToCylinder()
    axes.SetXAxisLabelText('')
    axes.SetYAxisLabelText('')
    axes.SetZAxisLabelText('')
    axes.SetTotalLength(1.5, 1.5, 1.5)
    axes.SetCylinderRadius(0.75*axes.GetCylinderRadius())
    axes.SetConeRadius(1.5 * axes.GetConeRadius())
    axes.SetSphereRadius(1.5 * axes.GetSphereRadius())
    return axes

def draw_unlit_line(ren, x1, y1, z1, x2, y2, z2, color=[0,0,0], width=1):
    line = vtk.vtkLineSource()
    line.SetPoint1(x1,y1,z1)
    line.SetPoint2(x2,y2,z2)

    # Create line mapper and actor
    linem = vtk.vtkPolyDataMapper()
    linem.SetInputConnection(line.GetOutputPort())
    linea = vtk.vtkActor()
    linea.SetMapper(linem)
    linea.GetProperty().SetLineWidth(width)
    linea.GetProperty().SetColor(color)
    linea.GetProperty().SetLighting(0)

    # Create tube mapper and actor (filtered line)
    tubef = vtk.vtkTubeFilter()
    tubef.SetInputConnection(line.GetOutputPort())
    tubef.SetRadius(0.05*width)
    tubef.SetNumberOfSides(50);
    tubef.CappingOn()
    tubef.Update()

    tubem = vtk.vtkPolyDataMapper()
    tubem.SetInputConnection(tubef.GetOutputPort())
    tubea = vtk.vtkActor()
    tubea.SetMapper(tubem)
    tubea.GetProperty().SetColor(color)
    tubea.GetProperty().SetLighting(0)
    ren.AddActor(tubea)

def draw_axes(ren, x, y, z):
    dx = x/10
    zz = 0#np.min([x, y, z])/100
    width = 0.2*np.min([x,y,z])
    draw_unlit_line(ren,-x/2+zz,-y/2+zz,-z/2+zz, -x/2+dx+zz,-y/2+zz,-z/2+zz, color=[1,0,0], width=width)
    draw_unlit_line(ren,-x/2+zz,-y/2+zz,-z/2+zz, -x/2+zz,-y/2+dx+zz,-z/2+zz, color=[0,1,0], width=width)
    draw_unlit_line(ren,-x/2+zz,-y/2+zz,-z/2+zz, -x/2+zz,-y/2+zz,-z/2+dx+zz, color=[0,0,1], width=width)

def draw_outer_box(ren, x, y, z, invert=False):
    if invert:
        color = [0.7, 0.7, 0.7]
    else:
        color = [0, 0, 0]
        
    width = 0.15*np.min([x,y,z])
    
    # Bottom layer
    draw_unlit_line(ren,-x/2,-y/2,-z/2, +x/2,-y/2,-z/2, color=color, width=width)
    draw_unlit_line(ren,+x/2,-y/2,-z/2, +x/2,+y/2,-z/2, color=color, width=width)
    draw_unlit_line(ren,+x/2,+y/2,-z/2, -x/2,+y/2,-z/2, color=color, width=width)
    draw_unlit_line(ren,-x/2,+y/2,-z/2, -x/2,-y/2,-z/2, color=color, width=width)

    # Top layer
    draw_unlit_line(ren,-x/2,-y/2,+z/2, +x/2,-y/2,+z/2, color=color, width=width)
    draw_unlit_line(ren,+x/2,-y/2,+z/2, +x/2,+y/2,+z/2, color=color, width=width)
    draw_unlit_line(ren,+x/2,+y/2,+z/2, -x/2,+y/2,+z/2, color=color, width=width)
    draw_unlit_line(ren,-x/2,+y/2,+z/2, -x/2,-y/2,+z/2, color=color, width=width)

    # Sides
    draw_unlit_line(ren,-x/2,-y/2,-z/2, -x/2,-y/2,+z/2, color=color, width=width)
    draw_unlit_line(ren,+x/2,-y/2,-z/2, +x/2,-y/2,+z/2, color=color, width=width)
    draw_unlit_line(ren,-x/2,+y/2,-z/2, -x/2,+y/2,+z/2, color=color, width=width)
    draw_unlit_line(ren,+x/2,+y/2,-z/2, +x/2,+y/2,+z/2, color=color, width=width)

def draw_origin_dot(ren):
    dot = vtk.vtkSphereSource()
    dot.SetRadius(.1)
    dotm = vtk.vtkPolyDataMapper()
    dotm.SetInputConnection(dot.GetOutputPort())
    dota = vtk.vtkActor()
    dota.SetMapper(dotm)
    dota.GetProperty().SetColor([0,0,0])
    ren.AddActor(dota)

def draw_double_arrow(ren, pos, direction):
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

        tp = utilsh.xyz2tp(*direction)
        arrowa.RotateWXYZ(-90, 0, 1, 0) # Align with Z axis
        arrowa.RotateWXYZ(np.rad2deg(tp[0]), 0, 1, 0)
        arrowa.RotateWXYZ(np.rad2deg(tp[1]), 0, 0, 1)
        arrowa.SetPosition(*pos)
        if i == 1:
            arrowa.RotateX(180)
            arrowa.RotateZ(180)

        ren.AddActor(arrowa)
        
# centers.shape = (M, 3)
# radii.shape = (M, N)
# M = number of ODFs and N = number of points on the sphere to plot
def draw_sphere_field(ren, centers, radii, plot_negative=True):
    M = radii.shape[0]
    N = radii.shape[1]
    vertices = utilsh.fibonacci_sphere(radii.shape[-1], xyz=True)
    faces = ConvexHull(vertices).simplices

    # Split into positive and negative
    if plot_negative:
        iradiis = [radii.clip(min=0), -radii.clip(max=0)]
    else:
        iradiis = [radii.clip(min=0)]

    # For both positive and negative
    for i, iradii in enumerate(iradiis):
        
        # Calculate vertices
        xyz_vertices = np.einsum('ij,ki->ikj', vertices, iradii) + centers
        all_xyz = xyz_vertices.reshape(-1, xyz_vertices.shape[-1], order='F') # Reshape
        all_xyz_vtk = numpy_support.numpy_to_vtk(all_xyz, deep=True) # Convert to vtk

        # Calculate faces
        all_faces = []
        for j in range(M):
            all_faces.append(faces + j*N)
        all_faces = np.concatenate(all_faces)
        all_faces = np.hstack((3 * np.ones((len(all_faces), 1)), all_faces))
        all_faces = np.ascontiguousarray(all_faces.ravel(), dtype='i8')
        all_faces_vtk = numpy_support.numpy_to_vtkIdTypeArray(all_faces, deep=True)

        # Populate points
        points = vtk.vtkPoints()
        points.SetData(all_xyz_vtk)
        
        # Calculate cells
        ncells = len(all_faces)
        cells = vtk.vtkCellArray()
        cells.SetCells(ncells, all_faces_vtk)

        # Populate polydata
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetPolys(cells)

        # Calculate colors        
        # TODO: Generalize colormaps
        cols = 255*np.ones(all_xyz.shape)
        iradiif = radii.flatten()
        if i == 0: # Red to white
            cols[:,1] = 255*(1-iradiif/(np.max(iradiif) + 1e-5))
            cols[:,2] = 255*(1-iradiif/(np.max(iradiif) + 1e-5))
        else: # Blue to white
            cols[:,0] = 255*(iradiif/(np.max(iradiif) + 1e-5))
            cols[:,1] = 255*(iradiif/(np.max(iradiif) + 1e-5))
        vtk_colors = numpy_support.numpy_to_vtk(cols, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        polydata.GetPointData().SetScalars(vtk_colors)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)

        # Create actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetAmbient(0.25)

        ren.AddActor(actor)
