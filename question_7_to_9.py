import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math

def question_7_warmup():
    """

    :return:
    """
    # Plot between -5 and 5 with .01 steps.
    x_axis = np.arange(-5, 5, 0.01)
    # Mean = 0, SD = 1.
    plt.plot(x_axis, norm.pdf(x_axis, 0, 1))
    x_axis = np.random.standard_normal(1000)

    plt.title("Histogram of normal dist Mean =0 Sigma =1")

    plt.hist(x_axis, density=True, histtype='stepfilled', alpha=0.5)
    plt.show()

def question_7_a(h_1, h_2):
    """

    :return:
    """
    fig, ax = plt.subplots()

    plt.title("Scatter plot of the points in the 2-D plane")

    ax.scatter(h_1[:,0], h_1[:,1], label='Mean = [0, (3/2)]')
    ax.scatter(h_2[:,0],h_2[:,1], label='Mean =[0, -(3/2)]')

    ax.legend()

    plt.show()

def question_7_c(h_1, h_2):
    """

    :return:
    """
    plt.title("Histogram for each of the two populations - x1")

    plt.hist(h_1[:, 0], density=True, histtype='stepfilled', alpha=0.5)
    plt.hist(h_2[:,0], density=True, histtype='stepfilled', alpha=0.5)

    plt.show()

    plt.title("Histogram for each of the two populations - x2")

    plt.hist(h_1[:, 1], density=True, histtype='stepfilled',alpha=0.5)
    plt.hist(h_2[:, 1], density=True, histtype='stepfilled',alpha=0.5)

    plt.show()

def question_7_d(h_1, h_2):
    """

    :return:
    """
    new_h_1 = []
    new_h_2 = []
    fig, ax = plt.subplots()

    for point in h_1:
        x,y = rotate_in_45_deg(point)
        new_point = np.array([x,y])
        new_h_1.append(new_point)

    for point in h_2:
        x,y = rotate_in_45_deg(point)
        new_point = np.array([x,y])
        new_h_2.append(new_point)

    new_h_1 = np.array(new_h_1)
    new_h_2 = np.array(new_h_2)

    plt.title("Scatter plot of the points in the 2-D plane rotated in 45 degrees")

    ax.scatter(new_h_1[:, 0], new_h_1[:, 1], label='Mean = [1,1]')
    ax.scatter(new_h_2[:, 0], new_h_2[:, 1], label='Mean = [-1,-1]')

    ax.legend()

    plt.show()

    question_7_c(new_h_1, new_h_2)

def rotate_in_45_deg(point):
    """

    :param point:
    :return:
    """
    x = point[0]
    y = point[1]
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)
    theta = theta + math.pi / 4
    x = r * math.cos(theta)
    y = r * math.sin(theta)

    return (x,y)


def question_7():
    """

    :return:
    """

    sigma = np.eye(2)
    mean_1 = np.array([1, 1])
    mean_2 = np.array([-1, -1])
    h_1 = np.random.multivariate_normal(mean_1, sigma, 1000)
    h_2 = np.random.multivariate_normal(mean_2, sigma, 1000)

    question_7_warmup()
    question_7_a(h_1, h_2)
    question_7_c(h_1, h_2)
    question_7_d(h_1, h_2)


def question_8():
    """

    :return:
    """
    V = np.array([[(1 / np.sqrt(2)), 1 / (np.sqrt(2))],
                  [(1 / np.sqrt(2)), -1 / (np.sqrt(2))]])
    V_T = np.transpose(V)
    sigma = [[2, 0], [0, 0.01]]
    cov = np.dot(np.dot(V, sigma), V_T)
    mean_1 = np.array([1, 1])
    mean_2 = np.array([-1, -1])
    h_1 = np.random.multivariate_normal(mean_1, cov, 1000)
    h_2 = np.random.multivariate_normal(mean_2, cov, 1000)

    question_7_a(h_1, h_2)
    question_7_c(h_1, h_2)

    sigma = [[0.01, 0], [0, 2]]
    cov = np.dot(np.dot(V, sigma), V_T)
    mean_1 = np.array([1, 1])
    mean_2 = np.array([-1, -1])
    h_1 = np.random.multivariate_normal(mean_1, cov, 1000)
    h_2 = np.random.multivariate_normal(mean_2, cov, 1000)

    question_7_a(h_1, h_2)
    question_7_c(h_1, h_2)


def question_9():
    """

    :return:
    """
    V = np.array([[(np.sqrt(3) / 2), (1 / 2)],
                  [(1 / 2), -np.sqrt(3) / (2)]])
    V_T = np.transpose(V)
    sigma = [[2, 0], [0, 0.01]]
    cov = np.dot(np.dot(V, sigma), V_T)
    mean_1 = np.array([0, (3/2)])
    mean_2 = np.array([0, -(3/2)])
    h_1 = np.random.multivariate_normal(mean_1, cov, 1000)
    h_2 = np.random.multivariate_normal(mean_2, cov, 1000)

    question_7_a(h_1, h_2)
    question_7_c(h_1, h_2)


if __name__ == '__main__':

    question_7()

    question_8()

    question_9()