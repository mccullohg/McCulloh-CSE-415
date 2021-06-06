"""run_3_class_4_feature_iris_data.py
NOTE: YOU SHOULD NOT NEED TO ADD ANY CODE TO THIS FILE.
HOWEVER, YOU MAY WISH TO MAKE MINOR EDITS IN ORDER TO
SEE DIFFERENT VIEWS OF THE DATA AND WEIGHT VECTORS.

Train a multiclass perceptron on 30 examples in the
full irises data set, and then test on the rest.

The classification and training should NOT be done in
this file, but you should implement those in the file
ternary_perceptron.py.
That program will be imported and called from here.

Version 1.1, Prashant Rangarajan and S. Tanimoto, May 11, 2021. Univ. of Washington.
"""

from ternary_perceptron import TernaryPerceptron  # The classifier and learning are here.
from plot_tp import PlotTernaryPerceptron
import csv  # For reading in data.
from matplotlib import pyplot as plt  # For plotting
import math  # For sqrt.


class PlotIrisTP(PlotTernaryPerceptron):
    """
    Plots the Ternary Perceptron after training it on the iris dataset
    with 4 features and 3 classes.
    """
    
    def __init__(self, tp, n_epochs, fts):
        super().__init__(tp, n_epochs, fts)  # Calls the constructor of the super class
        self.FEATURES = {0: 'Sepal Length',
                         1: 'Sepal Width - Feature 2',
                         2: 'Petal Length - Feature 3',
                         3: 'Petal Width'}  # Stores the names of the different features for Iris dataset
    
    def read_data(self):
        """
        Read data from the Iris dataset with 4 features and 2 classes
        for both training and testing.
        ---
        Overrides the method in PlotTernaryPerceptron
        """
        data_as_strings = list(csv.reader(open('iris-all-features-3-class-training.csv'), delimiter=','))
        self.TRAINING_DATA = [[float(f1), float(f2), float(f3), float(f4), int(c)]
                              for [f1, f2, f3, f4, c] in data_as_strings]
        data_as_strings = list(csv.reader(open('iris-all-features-3-class-testing.csv'), delimiter=','))
        self.TESTING_DATA = [[float(f1), float(f2), float(f3), float(f4), int(c)]
                             for [f1, f2, f3, f4, c] in data_as_strings]
    
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
        plt.xlabel(self.FEATURES[self.FEATURES_TO_PLOT[0]])
        plt.ylabel(self.FEATURES[self.FEATURES_TO_PLOT[1]])
        plt.title("Iris data with three weight vectors from multi-class perceptron training.")
        plt.show()


if __name__ == '__main__':
    ternary_perceptron = TernaryPerceptron(alpha=1.0)
    ptp = PlotIrisTP(ternary_perceptron, 100, fts=(0, 2))
    ptp.train()
    ptp.test()
    ptp.plot()
