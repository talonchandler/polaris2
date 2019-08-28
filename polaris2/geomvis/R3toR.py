import numpy as np
import vtk
from polaris2.geomvis import utilmpl, utilvtk

class xyz:
    def __init__(self, data, vox_dims=[1,1,1], vmin=0, vmax=None, xlabel='', title='', invert=True):
        self.data = data

        self.shape = data.shape*np.array(vox_dims)
        self.vox_dims = vox_dims
        self.invert = invert
        self.xlabel = utilmpl.shape2xlabel(self.shape)
        self.title = title
        self.vmin = vmin
        if vmax is None:
            self.vmax = np.max(data)
        else:
            self.vmax = vnmax

        # Setup renderer
        self.ren, self.renWin, self.iren = utilvtk.setup_render()

    def build_actors(self):
        # Draw 
        vol = np.interp(np.swapaxes(self.data, 0, 2), [self.vmin, self.vmax], [0, 255])
        vol = vol.astype('uint8')

        X, Y, Z = self.data.shape

        dataImporter = vtk.vtkImageImport()
        data_string = vol.tostring()
        dataImporter.CopyImportVoidPointer(data_string, len(data_string))
        dataImporter.SetDataScalarTypeToUnsignedChar()
        dataImporter.SetNumberOfScalarComponents(1)
        dataImporter.SetDataExtent(0, X-1, 0, Y-1, 0, Z-1)
        dataImporter.SetWholeExtent(0, X-1, 0, Y-1, 0, Z-1)

        # Create transfer mapping scalar value to opacity
        opacityTransferFunction = vtk.vtkPiecewiseFunction()
        opacityTransferFunction.AddPoint(0, 0)
        opacityTransferFunction.AddPoint(255, 1.0)
        
        # Create transfer mapping scalar value to color
        colorTransferFunction = vtk.vtkColorTransferFunction()
        colorTransferFunction.AddRGBPoint(255, 0.0, 0.0, 0.0)
        colorTransferFunction.AddRGBPoint(0, 1, 1, 1)
        if self.invert:
            colorTransferFunction.AddRGBPoint(0, 0.0, 0.0, 0.0)
            colorTransferFunction.AddRGBPoint(255, 1, 1, 1)

        # Describes how the data will look
        volumeProperty = vtk.vtkVolumeProperty()
        volumeProperty.SetColor(colorTransferFunction)
        volumeProperty.SetScalarOpacity(opacityTransferFunction)
        # volumeProperty.ShadeOn()
        # volumeProperty.SetInterpolationTypeToLinear()

        # The mapper / ray cast function know how to render the data
        volumeMapper = vtk.vtkGPUVolumeRayCastMapper()
        volumeMapper.SetBlendModeToMaximumIntensity()
        volumeMapper.SetSampleDistance(0.1)
        volumeMapper.SetAutoAdjustSampleDistances(0)
        volumeMapper.SetInputConnection(dataImporter.GetOutputPort())

        # The class vtkVolume is used to pair the preaviusly declared volume
        # as well as the properties to be used when rendering that volume.
        volume = vtk.vtkVolume()
        volume.SetMapper(volumeMapper)
        volume.SetProperty(volumeProperty)
        volume.SetPosition(*-self.shape/2)
        volume.SetScale(*self.vox_dims)

        if self.invert:
            self.ren.SetBackground([0,0,0])
        self.ren.AddActor(volume)

        # Draw extras
        utilvtk.draw_outer_box(self.ren, *self.shape, self.invert)
        utilvtk.draw_axes(self.ren, *self.shape)
        
        # Set cameras
        dist = 1.15*np.linalg.norm(self.shape)
        self.ren.GetActiveCamera().SetPosition(np.array([1,-1,1])*dist)
        self.ren.GetActiveCamera().SetViewUp([0,0,1])

    def increment_camera(self, az):
        self.ren.GetActiveCamera().Azimuth(az)
        
    def plot(self, f, fc, ss):
        ax = utilmpl.plot_template(f, fc, xlabel=self.xlabel, title=self.title,
                                   scale_bar=False, bump=1.2, invert=True)

        utilvtk.vtk2imshow(self.renWin, ax[0], ss)
        ax[0].axis('off')
        ax[1].axis('off')
