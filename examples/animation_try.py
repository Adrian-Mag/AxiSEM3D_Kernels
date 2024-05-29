from axikernels.core.handlers.element_output import ElementOutput

path = 'data/EXAMPLE_ELEMENT/output/elements'

element = ElementOutput(path)

# element.stream([6371000, 0, -20]).plot()
element.animation([0, 0, 0], [0, 0, 30])