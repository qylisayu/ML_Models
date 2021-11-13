from q3.check_grad import check_grad
from q3.utils import *
from q3.logistic import *

import matplotlib.pyplot as plt
import numpy as np


def run_logistic_regression():
    train_inputs, train_targets = load_train()
    #train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()

    N, M = train_inputs.shape

    #####################################################################
    # TODO:                                                             #
    # Set the hyperparameters for the learning rate, the number         #
    # of iterations, and the way in which you initialize the weights.   #
    #####################################################################
    hyperparameters = {
        "learning_rate": 0.2,
        "weight_regularization": 0.,
        "num_iterations": 600
    }
    weights = np.zeros((M + 1, 1))
    #weights = np.random.randn(M + 1, 1)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)

    # Begin learning with gradient descent
    #####################################################################
    # Modify this section to perform gradient descent, create plots,    #
    # and compute test error.                                           #
    #####################################################################

    loss_valid = []
    loss_train = []
    iterations = []
    for t in range(hyperparameters["num_iterations"]):
        f, df, y = logistic(weights, train_inputs, train_targets, hyperparameters)
        loss_train.append(f)
        ce, frac_correct = evaluate(valid_targets, logistic_predict(weights, valid_inputs))
        loss_valid.append(ce)
        iterations.append(t)
        weights = weights - (hyperparameters["learning_rate"] * df)

    plt.plot(iterations, loss_train, label='Train')
    plt.plot(iterations, loss_valid, label='Validation')
    plt.legend()
    plt.xlabel('number of iterations')
    plt.ylabel('CE loss')
    plt.title('CE Loss for Various Iterations - MNIST_TRAIN')
    plt.savefig('CE Loss for Various Iterations - MNIST_TRAIN.jpg')
    #plt.title('CE Loss for Various Iterations - MNIST_TRAIN_SMALL')
    #plt.savefig('CE Loss for Various Iterations - MNIST_TRAIN_SMALL.jpg')
    plt.show()

    print("\nhyperparameters: ", hyperparameters)
    y_train = logistic_predict(weights, train_inputs)
    ce_train, frac_correct_train = evaluate(train_targets, y_train)
    print("\nTrain_CE: " + str(ce_train), "\nTrain Classification Error: " + str(1. - frac_correct_train))

    y_valid = logistic_predict(weights, valid_inputs)
    ce_valid, frac_correct_valid = evaluate(valid_targets, y_valid)
    print("\nValidation_CE: " + str(ce_valid), "\nValidation Classification Error: " + str(1. - frac_correct_valid))

    test_inputs, test_targets = load_test()
    y_test = logistic_predict(weights, test_inputs)
    ce_test, frac_correct_test = evaluate(test_targets, y_test)
    print("\nTest_CE: " + str(ce_test), "\nTest Classification Error: " + str(1. - frac_correct_test))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def run_check_grad(hyperparameters):
    """ Performs gradient check on logistic function.
    :return: None
    """
    # This creates small random data with 20 examples and
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10
    #np.random.seed(123)
    weights = np.random.randn(num_dimensions + 1, 1)
    data = np.random.randn(num_examples, num_dimensions)
    targets = np.random.rand(num_examples, 1)

    diff = check_grad(logistic,
                      weights,
                      0.001,
                      data,
                      targets,
                      hyperparameters)

    print("diff =", diff)


if __name__ == "__main__":
    run_logistic_regression()
