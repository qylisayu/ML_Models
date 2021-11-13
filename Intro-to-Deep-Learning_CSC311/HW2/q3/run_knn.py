from q3.l2_distance import l2_distance
from q3.utils import *

import matplotlib.pyplot as plt
import numpy as np


def knn(k, train_data, train_labels, valid_data):
    """ Uses the supplied training inputs and labels to make
    predictions for validation data using the K-nearest neighbours
    algorithm.

    Note: N_TRAIN is the number of training examples,
          N_VALID is the number of validation examples,
          M is the number of features per example.

    :param k: The number of neighbours to use for classification
    of a validation example.
    :param train_data: N_TRAIN x M array of training data.
    :param train_labels: N_TRAIN x 1 vector of training labels
    corresponding to the examples in train_data (must be binary).
    :param valid_data: N_VALID x M array of data to
    predict classes for validation data.
    :return: N_VALID x 1 vector of predicted labels for
    the validation data.
    """
    dist = l2_distance(valid_data.T, train_data.T)
    nearest = np.argsort(dist, axis=1)[:, :k]

    train_labels = train_labels.reshape(-1)
    valid_labels = train_labels[nearest]

    # Note this only works for binary labels:
    valid_labels = (np.mean(valid_labels, axis=1) >= 0.5).astype(np.int)
    valid_labels = valid_labels.reshape(-1, 1)

    return valid_labels


def run_knn(classification_rate=None):
    train_inputs, train_targets = load_train()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()

    #####################################################################
    # Implement a function that runs kNN for different values of k,     #
    # plots the classification rate on the validation set, and etc.     #
    #####################################################################
    # Validation
    list_k = [1, 3, 5, 7, 9]
    rate = []
    for k in list_k:
        valid_labels = knn(k, train_inputs, train_targets, valid_inputs)
        num_correct = len(np.where(valid_targets == valid_labels)[0])
        classification_rate = float(num_correct)/float(len(valid_labels))
        rate.append(classification_rate)
        print('validation classification rate for k = ' + str(k) + ': ', classification_rate)

    plt.plot(list_k, rate)
    plt.xlabel('k')
    plt.ylabel('classification rate')
    plt.title('Classification Rate for Various k on Validation Data')
    plt.savefig('Classification Rate for Various k on Validation Data.jpg')
    plt.show()

    # Test
    test_k = [1, 3, 5]
    test_rate = []
    for k in test_k:
        test_labels = knn(k, train_inputs, train_targets, test_inputs)
        test_num_correct = len(np.where(test_targets == test_labels)[0])
        test_classification_rate = float(test_num_correct)/float(len(test_labels))
        test_rate.append(test_classification_rate)
        print('test classification rate for k = ' + str(k) + ': ', test_classification_rate)

    plt.clf()
    plt.plot(test_k, test_rate)
    plt.xlabel('k')
    plt.ylabel('classification rate')
    plt.title('Classification Rate for Various k on Test Data')
    plt.savefig('Classification Rate for Various k on Test Data.jpg')
    plt.show()

    return

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    run_knn()
