"""plot_tp.py
NOTE: YOU SHOULD NOT NEED TO ADD ANY CODE TO THIS FILE.
HOWEVER, YOU MAY WISH TO MAKE MINOR EDITS IN ORDER TO
SEE DIFFERENT VIEWS OF THE DATA AND WEIGHT VECTORS.

Implements a class that can train and plot a ternary perceptron
for any dataset with classes 0, 1 and 2, picking a subset of features
to plot in 2D.

Version 1.1, Prashant Rangarajan and S. Tanimoto, May 11, 2021. Univ. of Washington.
"""

from ternary_perceptron import TernaryPerceptron  # The classifer and learning are here.
import csv  # For reading in data.
from matplotlib import pyplot as plt  # For plotting
import math  # For sqrt.


class PlotTernaryPerceptron:
    """
    Plots the Ternary Perceptron on the given dataset with 3 classes.
    """
    
    def __init__(self, tp, n_epochs, fts=(0, 1)):
        """
        Initializes the class
        ---
        X_MIN, X_MAX, Y_MIN, Y_MAX: Bounding coordinate of the plot
        TRAINING_DATA: To be filled with input data on which the model is trained/plotted
        TESTING_DATA: Can test the perceptron using separate test data (if required)
        MAX_EPOCHS: Maximum number of epochs the perceptron runs for.
        PLOTLINE_COUNT: Keeps track of epoch numbers of intermediate plot separators
        tp: Input Ternary Perceptron
        FEATURES_TO_PLOT: Selects which 2 features to plot in 2D. If the data has only
                          2 features, [0,1] can be used as a default.
        """
        self.X_MIN = 0
        self.X_MAX = 0
        self.Y_MIN = 0
        self.Y_MAX = 0
        self.TRAINING_DATA = []
        self.TESTING_DATA = []
        self.PLOTLINE_COUNT = 1
        self.MAX_EPOCHS = n_epochs
        self.tp = tp
        self.FEATURES_TO_PLOT = fts
    
    def read_data(self):
        """
        Read training data from the given dataset
        Also reads testing data if necessary
        ---
        Contains a placeholder train dataset
        """
        self.TRAINING_DATA = [
            [8, -8, 2, 9, 2],
            [8, -9, 0, -5, 1],
            [8, -4, 6, 5, 2],
            [-4, -5, -3, 1, 0],
            [6, 3, 9, -3, 1],
            [-3, 3, -10, -1, 0],
            [4, -4, 8, 5, 2],
            [-8, -7, -7, -3, 0],
            [-8, -2, -9, -10, 0],
            [-3, -7, 0, 3, 0]]
    
    def plot_2d_points(self, points_to_plot):
        """
        points_to_plot: list of triples of the form [xi, yi, ci]
        where ci is either 0, 1, or 2.
        """
        xpoints = [pt[0] for pt in points_to_plot]
        self.X_MIN = min(xpoints)
        self.X_MAX = max(xpoints)
        plt.figure(figsize=(10, 6))
        ypoints = [pt[1] for pt in points_to_plot]
        self.Y_MIN = min(ypoints)
        self.Y_MAX = max(ypoints)
        markers = [self.get_marker_style(pt[-1]) for pt in
                   points_to_plot]  # depending on iris category.
        for (x, y, c) in zip(xpoints, ypoints, markers):
            plt.plot(x, y, c, linestyle='')
    
    @staticmethod
    def get_marker_style(c):
        return ['o b', 'P g', 's r'][c]  # blue circles, green pluses, red squares.
    
    def plot_weight_vectors(self, W):
        """Add to the plot so far three vectors that best represents
        the current sets of weights in the directions that have been
        chosen for visualization.
        
        Show each vector as emanating from a common starting point.
        It's not really necessary here, but this code scales the vectors;
        it might be useful if more control of the plot is desired.
        """
        X_MIDDLE = 0.5 * (self.X_MIN + self.X_MAX)
        Y_MIDDLE = 0.5 * (self.Y_MIN + self.Y_MAX)
        V = [[W[c][self.FEATURES_TO_PLOT[0]], W[c][self.FEATURES_TO_PLOT[1]]] for c in range(3)]
        lengths_sq = [(V[c][0]) ** 2 + (V[c][1]) ** 2 for c in range(3)]
        lengths = [math.sqrt(m) for m in lengths_sq]
        max_len = max(lengths)
        if max_len == 0:
            max_len = 1.0
        scale = 5 / max_len
        v_scaled = [[vi * scale for vi in v] for v in V]
        arrowhead_xs = [v[0] for v in v_scaled]
        arrowhead_ys = [v[1] for v in v_scaled]
        plt.quiver([X_MIDDLE] * 3, [Y_MIDDLE] * 3, arrowhead_xs, arrowhead_ys, color=['b', 'g', 'r'], scale=21)
    
    def train(self, verbose=False):
        """
        Trains the Ternary perceptron
        verbose: If True, prints out the weights and changed count
                at every epoch
        """
        self.read_data()
        for i in range(self.MAX_EPOCHS):
            changed_count = self.tp.train_for_an_epoch(self.TRAINING_DATA)
            if changed_count == 0:
                print("Converged in ", i, " epochs.")
                print("TRAINING IS DONE")
                return
            if verbose:
                print(f"\nchanged_count= {changed_count}")
                print("Weights:")
                print(*self.tp.W, sep='\n')
        print(f"Training did not converge in {self.MAX_EPOCHS} epochs.")
    
    def test(self):
        """
        If we have testing data, the child class will implement this method
        """
        pass
    
    def plot(self):
        """
        Plots the dataset as well as the ternary classifier
        """
        points_to_plot = [[I[self.FEATURES_TO_PLOT[0]], I[self.FEATURES_TO_PLOT[1]], I[-1]] for I in self.TRAINING_DATA]
        self.plot_2d_points(points_to_plot)
        self.plot_weight_vectors(self.tp.W)
        plt.show()


if __name__ == '__main__':
    ternary_perceptron = TernaryPerceptron(alpha=1.0)
    ptp = PlotTernaryPerceptron(ternary_perceptron, 100, (2, 3))
    ptp.train(verbose=True)
    ptp.test()
    ptp.plot()
