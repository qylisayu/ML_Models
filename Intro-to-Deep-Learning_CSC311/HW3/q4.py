'''
Question 4 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''
from scipy.special import logsumexp

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    # Compute means
    for k in range(10):
        means[k] = np.mean(data.get_digits_by_label(train_data, train_labels, k), axis=0)
    return means

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class
    '''
    covariances = np.zeros((10, 64, 64))
    # Compute covariances
    means = compute_mean_mles(train_data, train_labels)
    for k in range(10):
        data_k = data.get_digits_by_label(train_data, train_labels, k)
        diff = data_k - means[k]
        prod = diff.T @ diff
        covariances[k] = prod/float(len(data_k))
    return covariances

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array
    '''
    n = len(digits)
    log_likelihood = np.zeros((n, 10))
    for k in range(10):
        sigma = covariances[k] + 0.01 * np.identity(64)
        for i in range(n):
            diff = (digits[i] - means[k])
            prod = diff.T @ np.linalg.inv(sigma) @ diff
            t3 = 0 - prod/2.
            t1 = 0 - np.log(2. * np.pi) * 32
            t2 = 0 - np.log(np.linalg.det(sigma))/2.
            log_likelihood[i, k] = t1 + t2 + t3
    return log_likelihood


def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    gen_like = generative_likelihood(digits, means, covariances)
    cond_like = np.zeros((len(digits), 10))
    for i in range(len(digits)):
        cond_like[i] = gen_like[i] - logsumexp(gen_like[i])
    #cond_like = gen_like - (gen_like.sum(axis=1)).reshape((-1, 1))
    return cond_like

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute as described above and return
    lst = cond_likelihood[np.arange(len(cond_likelihood)), labels.astype(int)]
    avg = np.mean(lst)
    return avg

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    pred = np.argmax(cond_likelihood, axis=1)
    return pred

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)

    # Evaluation
    train_avg_cond_like = avg_conditional_likelihood(train_data, train_labels, means, covariances)
    test_avg_cond_like = avg_conditional_likelihood(test_data, test_labels, means, covariances)
    print("Avg conditional log-likelihood for train: " + str(train_avg_cond_like))
    print("Avg conditional log-likelihood for test: " + str(test_avg_cond_like))
    y_train = classify_data(train_data, means, covariances)
    y_test = classify_data(test_data, means, covariances)
    train_acc = np.mean(y_train == train_labels)
    test_acc = np.mean(y_test == test_labels)
    print("Training accuracy: " + str(train_acc*100.) + "%")
    print("Testing accuracy: " + str(test_acc*100.) + "%")

    # Plot
    eig_vecs = []
    for i in range(10):
        eig_val, eig_vec = np.linalg.eig(covariances[i])
        max_i = np.argmax(eig_val)
        eig_vecs.append(eig_vec[:, max_i].reshape(8, 8))
    concat = np.concatenate(eig_vecs, axis=1)
    plt.imshow(concat, cmap="gray")
    plt.show()



if __name__ == '__main__':
    main()
