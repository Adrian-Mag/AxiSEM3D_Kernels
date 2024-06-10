from axikernels.aux.mesher import SliceMesh, SphereMesh
import matplotlib.pyplot as plt


new_mesh = SphereMesh('shell_Kd_400_P.h5')
new_mesh.plot_on_mesh(remove_outliers=True, gamma=0.1)