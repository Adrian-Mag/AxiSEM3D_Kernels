from axikernels.core.handlers.element_output import ElementOutput
from axikernels.core.handlers.station_output import StationOutput

a = ElementOutput('data/1D_KERNEL_EXAMPLE_20s/output/elements')
a.obspyfy('data/1D_KERNEL_EXAMPLE/input/STA_10DEG_GRID.txt', channels=['UZ', 'UR', 'UT'])

""" a = StationOutput('data/1D_KERNEL_EXAMPLE_20s/output/stations/Station_grid')
a.obspyfy()
 """