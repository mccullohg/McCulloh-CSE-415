"""run_ring_data.py

Gordon McCulloh

Extends the Class PlotBinaryPerceptron

Follows model of run_2_class_2_feature_iris_data.py
Version 1.1, Prashant Rangarajan and S. Tanimoto, May 11, 2021. Univ. of Washington.
"""

from binary_perceptron import BinaryPerceptron  # Your implementation of binary perceptron
from plot_bp import PlotBinaryPerceptron
import csv  # For loading data.
from matplotlib import pyplot as plt  # For creating plots.
from remapper import remap


class PlotRingBP(PlotBinaryPerceptron):
    """
    Plots the Binary Perceptron after training it on the Ring dataset
    ---
    Extends the class PlotBinaryPerceptron
    """

    def __init__(self, bp, plot_all=True, n_epochs=25):
        super().__init__(bp, plot_all, n_epochs)  # Calls the constructor of the super class

    def read_data(self):
        """
        Read data from the Ring dataset with 2 features and 2 classes
        for both training and testing.
        ---
        Overrides the method in PlotBinaryPerceptron
        """
        IS_REMAPPED = 1
        if IS_REMAPPED:
            data_as_strings = list(csv.reader(open('ring-data.csv'), delimiter=','))
            self.TRAINING_DATA = [[remap(float(f1),float(f2))[0],remap(float(f1),float(f2))[1],\
                                   int(c)] for [f1, f2, c] in data_as_strings]
            data_as_strings = list(csv.reader(open('ring-data.csv'), delimiter=','))
            self.TESTING_DATA = [[remap(float(f1),float(f2))[0],remap(float(f1),float(f2))[1],\
                                  int(c)] for [f1, f2, c] in data_as_strings]
        else:
            data_as_strings = list(csv.reader(open('ring-data.csv'), delimiter=','))
            self.TRAINING_DATA = [[float(f1), float(f2), int(c)] for [f1, f2, c] in data_as_strings]
            data_as_strings = list(csv.reader(open('ring-data.csv'), delimiter=','))
            self.TESTING_DATA = [[float(f1), float(f2), int(c)] for [f1, f2, c] in data_as_strings]

    def test(self):
        """
        Evaluates the Binary Perceptron on the test set.
        Prints out the number of errors.
        """
        error_count = 0
        N_TESTING = len(self.TESTING_DATA)
        for i in range(N_TESTING):
            x_vec = self.TESTING_DATA[i][:-1]
            y = self.TESTING_DATA[i][-1]

            result = self.bp.classify(x_vec)
            if result != y: error_count += 1
        print(error_count, " errors on the test data, out of ", N_TESTING, "items.")

    def plot(self):
        """
        Plots the dataset as well as the binary classifier
        ---
        Overrides the method in PlotBinaryPreceptron
        """
        plt.title("Iris setosa (blue) vs iris versicolor (red)")
        plt.xlabel("Sepal length")
        plt.ylabel("Petal length")
        plt.legend(loc='best')
        plt.show()


if __name__ == '__main__':
    binary_perceptron = BinaryPerceptron(alpha=0.5)
    pbp = PlotRingBP(binary_perceptron)
    pbp.train()
    pbp.test()
    pbp.plot()

