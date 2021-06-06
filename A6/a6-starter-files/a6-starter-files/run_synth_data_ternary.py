"""run_synth_data_ternary.py

Modeled after run_3_class_4_feature_iris_data.py
Version 1.1, Prashant Rangarajan and S. Tanimoto, May 11, 2021. Univ. of Washington.

Gordon McCulloh, CSE 415, Sp21
"""

from ternary_perceptron import TernaryPerceptron  # The classifier and learning are here.
from plot_tp import PlotTernaryPerceptron
import csv  # For reading in data.
from matplotlib import pyplot as plt  # For plotting
import math  # For sqrt.


class PlotMultiTP(PlotTernaryPerceptron):
    """
    Plots the Ternary Perceptron after training it on the synthetic dataset
    with 2 features and 3 classes.
    """

    def __init__(self, tp, n_epochs, fts):
        super().__init__(tp, n_epochs, fts)  # Calls the constructor of the super class
        self.FEATURES = {0: 'Feature 1',
                         1: 'Feature 2'}  # Stores the names of the different features

    def read_data(self):
        """
        Read data from the Synthetic dataset with 2 features and 3 classes
        for both training and testing.
        ---
        Overrides the method in PlotTernaryPerceptron
        """
        data_as_strings = list(csv.reader(open('synthetic_data.csv'), delimiter=','))
        self.TRAINING_DATA = [[float(f1), float(f2), int(c)]
                              for [f1, f2, c] in data_as_strings]
        data_as_strings = list(csv.reader(open('synthetic_data.csv'), delimiter=','))
        self.TESTING_DATA = [[float(f1), float(f2), int(c)]
                             for [f1, f2, c] in data_as_strings]

    def test(self):
        """
        Evaluates the Ternary Perceptron on the test set
        and prints out the number of errors.
        """
        error_count = 0
        N_TESTING = len(self.TESTING_DATA)
        for i in range(N_TESTING):
            x_vec = self.TESTING_DATA[i][:-1]
            y = self.TESTING_DATA[i][-1]
            result = self.tp.classify(x_vec)
            if result != y:
                error_count += 1
        print(error_count, " errors on the test data, out of ", N_TESTING, "items.")

    def plot(self):
        """
        Plots the dataset as well as the ternary classifier
        """
        points_to_plot = [[I[self.FEATURES_TO_PLOT[0]], I[self.FEATURES_TO_PLOT[1]], I[-1]] for I in self.TRAINING_DATA]
        self.plot_2d_points(points_to_plot)
        self.plot_weight_vectors(self.tp.W)
        plt.xlabel(" ")
        plt.ylabel(" ")
        plt.title("Synthetic data with three weight vectors from multi-class perceptron training.")
        plt.show()


if __name__ == '__main__':
    ternary_perceptron = TernaryPerceptron(alpha=0.5)
    ptp = PlotMultiTP(ternary_perceptron, 50, fts=(0, 1))
    ptp.train()
    ptp.test()
    ptp.plot()
