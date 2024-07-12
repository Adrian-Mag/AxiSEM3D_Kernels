from axikernels.aux import mesher
shell = mesher.SphereMesh(radius=5000000, n=50000, domain=[-20, 20, -10, 60])
shell.plot()