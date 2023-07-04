from AxiSEM3D_Data_Handler.element_output import ElementOutput

path = '/disks/data/PhD/CMB/simu1D_element/BACKWARD_DATA/output/elements/entire_earth'

element = ElementOutput(path)

# element.stream([6371000, 0, -20]).plot()
element.animation([0, 0, 0], [0, 0, 30], resolution=300, R_min=3400000, paralel_processing=True)