"""run_synth_data_1_vs_all

Extends the Class PlotBinaryPerceptron

Modeled after run_2_class_2_feature_iris_data.py
Version 1.1, Prashant Rangarajan and S. Tanimoto, May 11, 2021. Univ. of Washington.

Gordon McCulloh, CSE 415, Sp21
"""

from binary_perceptron import BinaryPerceptron # Your implementation of binary perceptron
from plot_bp import PlotBinaryPerceptron
import csv  # For loading data.
from matplotlib import pyplot as plt  # For creating plots.


class PlotMultiBPOneVsAll(PlotBinaryPerceptron):
    """
    Plots the Binary Perceptron after training it on the synthetic dataset
    ---
    Extends the class PlotBinaryPerceptron
    """

    def __init__(self, bp, plot_all=False, n_epochs=50):
        super().__init__(bp, plot_all, n_epochs)  # Calls the constructor of the super class

    def read_data(self):
        """
        Read data from the Synthetic dataset with 2 features and 2 classes
        for both training and testing.
        ---
        Overrides the method in PlotBinaryPerceptron
        """
        POSITIVE = 2  # create instance variable to choose isolated class
        data_as_strings = list(csv.reader(open('synthetic_data.csv'), delimiter=','))
        self.TRAINING_DATA = []
        for [f1, f2, c] in data_as_strings:
            if int(c) == POSITIVE:
                xy_pos = [float(f1), float(f2), 1.0]
                self.TRAINING_DATA.append(xy_pos)
            else:
                xy_neg = [float(f1), float(f2), -1.0]
                self.TRAINING_DATA.append(xy_neg)
        data_as_strings = list(csv.reader(open('synthetic_data.csv'), delimiter=','))
        self.TESTING_DATA = []
        for [f1, f2, c] in data_as_strings:
            if int(c) == POSITIVE:
                xy_pos = [float(f1), float(f2), 1.0]
                self.TESTING_DATA.append(xy_pos)
            else:
                xy_neg = [float(f1), float(f2), -1.0]
                self.TESTING_DATA.append(xy_neg)

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
        plt.title("Plot with class 2 as the positive class")
        plt.xlabel(" ")
        plt.ylabel(" ")
        plt.legend(loc='best')
        plt.show()



if __name__=='__main__':
    binary_perceptron = BinaryPerceptron(alpha=0.5)
    pbp = PlotMultiBPOneVsAll(binary_perceptron)
    pbp.train()
    pbp.test()
    pbp.plot()
