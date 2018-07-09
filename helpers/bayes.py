import numpy
import pandas
from scipy.stats import norm
import matplotlib.pyplot as plot
from sklearn.datasets import make_blobs

from matplotlib import cm
cmap = cm.get_cmap('rainbow')


def to_dataframe(observations, categories):
    """
    Convert observations/categories values into a DataFrame in a nice way.
    """
    df = pandas.DataFrame()
    df['x0'] = observations[:, 0]
    df['x1'] = observations[:, 1]
    df['y'] = categories
    return df


def joint_histograms(observations, categories):
    """
    Plot 2D data with histograms of each category in each dimension.
    """
    df = to_dataframe(observations, categories)
    fig = plot.figure()
    classes = numpy.unique(categories).size
    if classes == 1: # hack around the no-classes case
        classes = 2

    # adapted from https://matplotlib.org/examples/pylab_examples/scatter_hist.html
    from matplotlib.ticker import NullFormatter
    nullfmt = NullFormatter()

    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    # start with a rectangular Figure
    #plt.figure(1, figsize=(8, 8))

    axScatter = plot.axes(rect_scatter)
    axHistx = plot.axes(rect_histx)
    axHisty = plot.axes(rect_histy)

    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHistx.yaxis.set_major_formatter(nullfmt)
    axHisty.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)
    
    # the scatter plot:
    axScatter.scatter(df['x0'], df['x1'], c=categories/(classes-1.0), cmap='rainbow', edgecolor='k')

    # now determine nice limits by hand:
    binwidth = 0.5
    xymax = numpy.max([numpy.max(numpy.fabs(df['x0'])), numpy.max(numpy.fabs(df['x1']))])
    lim = (int(xymax/binwidth) + 1) * binwidth
    xmin, xmax = df['x0'].min()-1, df['x0'].max()+1
    ymin, ymax = df['x1'].min()-1, df['x1'].max()+1

    axScatter.set_xlim((xmin, xmax))
    axScatter.set_ylim((ymin, ymax))

    xbins = numpy.arange(xmin, xmax, binwidth)
    ybins = numpy.arange(ymin, ymax, binwidth)
    for cat in numpy.unique(categories):
        clr = cmap(cat/(classes-1.0))
        axHistx.hist(df['x0'][df['y'] == cat], bins=xbins, color=clr)
        axHisty.hist(df['x1'][df['y'] == cat], bins=ybins, color=clr, orientation='horizontal')

    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())


def pdf_2d(observations, categories, category):
    """
    Plot a normal probability density function for one category of this labeled data.
    """
    df = to_dataframe(observations, categories)
    one_cat = df[df['y'] == category]
    mean_x = one_cat['x0'].mean()
    mean_y = one_cat['x1'].mean()
    stddev_x = one_cat['x0'].std()
    stddev_y = one_cat['x1'].std()
    x_min, x_max = df['x0'].min(), df['x0'].max()
    y_min, y_max = df['x1'].min(), df['x1'].max()
    xg = numpy.linspace(x_min, x_max, 100)
    yg = numpy.linspace(y_min, y_max, 100)
    xx, yy = numpy.meshgrid(xg, yg)

    pdf = norm.pdf(xx, mean_x, stddev_x) * norm.pdf(yy, mean_y, stddev_y)

    plot.imshow(numpy.flip(pdf, axis=0), extent=[x_min, x_max, y_min, y_max], aspect='auto')


def plot_decision(model, X, y=None, width=400, height=400):
    """
    Plot the decision boundaries of this model on the X values.
    
    Assumes >= 2 features. Plots first as x axis; second as y.
    """
    x0 = X[:, 0]
    y0 = X[:, 1]
    xg = numpy.linspace(x0.min(), x0.max(), width)
    yg = numpy.linspace(y0.min(), y0.max(), height)
    xx, yy = numpy.meshgrid(xg, yg)
    X_grid = numpy.vstack([xx.ravel(), yy.ravel()]).T
    y_grid = model.predict(X_grid)
    plot.contourf(xx, yy, y_grid.reshape((height, width)), cmap=cmap)
    if y is not None:
        plot.scatter(x0, y0, c=y, cmap=cmap, edgecolor='k')


def sample_data_1():
    return make_blobs(n_samples=500, n_features=2, centers=3, cluster_std=3.0, random_state=10)


def prediction_row(X_row, model, city_index):
    label = X_row['label']
    X_row = X_row.drop('label')
    pred = model.predict([X_row])[0]
    proba = model.predict_proba([X_row])[0]
    return {
        'label': label,
        'prediction': pred,
        'label_proba': proba[city_index[label]],
        'pred_proba': proba[city_index[pred]],
    }


def incorrect_predictions(model, X_test, y_test):
    """
    Incorrect predictions on this data, with GaussianNB's probabilities for those classes.
    """
    city_index = {city: pos for pos, city in enumerate(model.classes_)}
    combined_test = X_test.copy()
    combined_test['label'] = y_test
    incorrect_predictions = combined_test.apply(prediction_row, result_type='expand',
        axis=1, args=(model, city_index))
    incorrect_predictions = incorrect_predictions[incorrect_predictions['label'] != incorrect_predictions['prediction']]
    return incorrect_predictions


def correlated_data():
    n = 400
    df_correlated = pandas.DataFrame({
        'x0': numpy.concatenate((
            numpy.random.normal(-3, 1.5, n),
            numpy.random.normal(3, 1.5, n),
        )),
        'x1': numpy.concatenate((
            numpy.random.normal(0, 3, n),
            numpy.random.normal(0, 3, n),
        )),
        'y': numpy.concatenate((
            numpy.zeros((n,)),
            numpy.ones((n,)),
        )),
    })
    df_correlated['x0'] = df_correlated['x0'] + 1.5 * df_correlated['x1']

    X_correlated = df_correlated[['x0', 'x1']].values
    y_correlated = df_correlated['y'].values
    return X_correlated, y_correlated
