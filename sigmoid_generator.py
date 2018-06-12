
"""
forever:
    I give five parameters, the program uses it to plot sigmoid
    x = [1, ..., 30]
"""

import matplotlib.pyplot as plt


def plot_sigmoid(model, params, length):

    x_values = list(range(1, length+1))
    series = [model.predict(params, x_t) for x_t in x_values]

    plt.plot(x_values, series)
    plt.show()
