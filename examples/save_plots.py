from axikernels.aux.mesher import SliceMesh, SphereMesh
import matplotlib.pyplot as plt

new_mesh = SliceMesh('vp_3D_30s_P.h5')
new_mesh.plot_on_mesh(cbar_range=[-5e-20, 5e-20], filename='vp_1D_30s_P.png')

""" new_mesh = SphereMesh('shell_Kd_400km_3D_30s_P_2.h5')
new_mesh.plot_on_mesh(remove_outliers=True, cbar_range=[-1e-18, 1e-18]) """