import numpy
import matplotlib.pyplot as plot
from skimage.color import lab2rgb, hsv2rgb, rgb2lab, rgb2hsv

# representative RGB colours for each label, for nice display
COLOUR_RGB = {
    'red': (255, 0, 0),
    'orange': (255, 114, 0),
    'yellow': (255, 255, 0),
    'green': (0, 230, 0),
    'blue': (0, 0, 255),
    'purple': (187, 0, 187),
    'brown': (117, 60, 0),
    'pink': (255, 187, 187),
    'black': (0, 0, 0),
    'grey': (150, 150, 150),
    'white': (255, 255, 255),
}
name_to_rgb = numpy.vectorize(COLOUR_RGB.get, otypes=[numpy.uint8, numpy.uint8, numpy.uint8])

def plot_predictions(model, value=1.0, resolution=512):
    """
    Create a slice of HSV colour space with given value; predict with the model;
    plot the results.
    """
    wid = resolution
    hei = resolution
    n_ticks = 5

    # create a hei*wid grid of HSV colour values, with V=value
    hg = numpy.linspace(0, 1, wid)
    sg = numpy.linspace(0, 1, hei)
    hh, ss = numpy.meshgrid(hg, sg)
    vv = value * numpy.ones((hei, wid))
    hsv_grid = numpy.stack([ss, hh, vv], axis=2)

    # convert to RGB for consistency with original input
    X_grid = hsv2rgb(hsv_grid)

    # predict and convert predictions to colours so we can see what's happening
    y_grid = model.predict(X_grid.reshape((wid*hei, 3)))
    pixels = numpy.stack(name_to_rgb(y_grid), axis=1) / 255.0
    pixels = pixels.reshape((hei, wid, 3))

    # plot input and predictions
    plot.figure(figsize=(10, 5))
    plot.suptitle('Predictions at V=%g' % (value,))
    plot.subplot(1, 2, 1)
    plot.title('Inputs')
    plot.xticks(numpy.linspace(0, wid, n_ticks), numpy.linspace(0, 1, n_ticks))
    plot.yticks(numpy.linspace(0, hei, n_ticks), numpy.linspace(0, 1, n_ticks))
    plot.xlabel('S')
    plot.ylabel('H')
    plot.imshow(X_grid.reshape((hei, wid, 3)))

    plot.subplot(1, 2, 2)
    plot.title('Predicted Labels')
    plot.xticks(numpy.linspace(0, wid, n_ticks), numpy.linspace(0, 1, n_ticks))
    plot.yticks(numpy.linspace(0, hei, n_ticks), numpy.linspace(0, 1, n_ticks))
    plot.xlabel('S')
    plot.imshow(pixels)
