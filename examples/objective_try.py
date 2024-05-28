from ..core.kernels.objective_function import L2Objective_Function
from ..core.handlers.element_output import ElementOutput
from ..core.handlers.station_output import StationOutput
from ..core.handlers.obspy_output import ObspyfiedOutput


fwd = ElementOutput('/disks/data/PhD/CMB/simu1D_element/FORWARD_DATA/output/elements/entire_earth')
real = ObspyfiedOutput('/disks/data/PhD/CMB/simu3D_CMB/REAL_DATA/output/stations/Station_grid/obspyfied')
obj = L2Objective_Function(forward_data=fwd, real_data=real,
                           window_left=None, window_right=None)
fwd.stream([6371000, 0, 30], ['U']).plot()
real.stream.select(station='22').plot()
obj.evaluate_objective_function(network='A', station='22', location='*')
