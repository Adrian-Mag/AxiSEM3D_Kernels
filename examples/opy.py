from axikernels.core.handlers.element_output import ElementOutput

a = ElementOutput('data/1D_KERNEL_EXAMPLE/output/elements')
a.obspyfy('data/1D_KERNEL_EXAMPLE/input/STA_10DEG_GRID.txt', channels=['UZ', 'UR', 'UT'])