from ..core.handlers.element_output import ElementOutput

path = '/disks/data/PhD/CMB/simu1D_element/BACKWARD_DATA/output/elements/entire_earth'
element = ElementOutput(path_to_element_output=path)
element.animation([0,0,0], [0,0,30])