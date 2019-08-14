import numpy as np
from polaris2.micro import det, ill

class Microscope:
    """
    A Microscope represents an experiment that collects a single frame of 
    intensity data.  

    A Microscope is specified by its illumination path (an Illuminator object),
    and its detection path (a Detector object).
    """
    def __init__(self, ill=ill.WideField(), det=det.FourF(),
                 color=(0,1,0.3), spang_coupling=True):
        self.ill = ill
        self.det = det

