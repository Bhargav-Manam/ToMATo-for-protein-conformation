import matplotlib.pyplot as plt
import numpy as np


def get_scatterplot(coordinates: np.ndarray, title: str = '', xlabel: str = 'Dimension1', ylabel: str = 'Dimension2',
                    c=None):
    """
        Create a scatter plot of the given coordinates.

        Args:
            coordinates: A numpy array of shape (n, 2) representing the (x, y)
                coordinates of n data points.
            title (optional): The title of the plot.
            xlabel (optional): The label for the x-axis.
            ylabel (optional): The label for the y-axis.
            c (optional): A color or sequence of colors for the data points.

        Returns:
            The matplotlib plot object.
    """
    plt.scatter(coordinates[:, 0], coordinates[:, 1], s=0.5, c=c)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return plt
