from axikernels.core.handlers import station_output
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

my_output = station_output.StationOutput('data/EXAMPLE/output/stations/Station_grid')
data = my_output.stream(networks=['II'], station_names=['CMLA'], locations=[''], channels=['UZ'])
print(data)
data.plot()
plt.show()