import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


def question_10_b_c(set_d, set_t, sigma):
    """

    :param set_d:
    :param set_t:
    :return:
    """

    set_s = set_d[:500, 0]
    labels_s = set_d[:500, 1]
    set_v = set_d[500:, 0]
    labels_v = set_d[500:, 1]

    H = []

    for d in range(1, 16):
        p = np.polyfit(set_s, labels_s, d)
        H.append(p)

    error = []
    validation_error = []

    for index, h in enumerate(H):
        mse = calculate_mse(set_s, labels_s, h)
        error.append(mse)
        mse = calculate_mse(set_v, labels_v, h)
        validation_error.append(mse)
    h_star = np.argmin(validation_error)
    print(h_star)
    print(calculate_mse(set_v, labels_v, H[h_star]))
    print(calculate_mse(set_t[:, 0], set_t[:, 1], H[h_star]))

    plt.title("k=2 Validation error plot, sigma=%d" %sigma)
    plt.xlabel("Mean error")
    plt.ylabel("Degree")
    plt.plot(range(1, 16), validation_error)

    plt.show()


def question_10_d(set_d, set_t, k, sigma):
    """

    :param set_d:
    :param set_t:
    :param k:
    :return:
    """
    fig, ax = plt.subplots()

    kf = KFold(n_splits=k)

    error_d = []
    validation_error = []
    H = []

    for d in range(1, 16):

        sum_d = 0
        val_sum_d = 0

        for training_indexes, testing_indexes in kf.split(set_d):
            train_set, test_set = set_d[:, 0][training_indexes], set_d[:, 0][
                testing_indexes]
            train_labels, test_labels = set_d[:, 1][training_indexes], set_d[:, 1][
                testing_indexes]

            p = np.poly1d(np.polyfit(train_set, train_labels, d))
            H.append(p)
            sum_d += (calculate_mse(train_set, train_labels, p))

            val_sum_d += (calculate_mse(test_set, test_labels, p))

        error_d.append(1 / 5 * (sum_d))
        validation_error.append(1 / 5 * (val_sum_d))


    plt.title("k=5 degree Mean error plot with sigma=%d" %sigma)
    plt.xlabel("Mean error")
    plt.ylabel("Degree")

    ax.plot(range(15), error_d, color="blue", label="Training error")

    ax.plot(range(15), validation_error, color="orange",
             label="Validation error")
    ax.legend()
    plt.show()

    # h_star = np.argmin(validation_error)
    # print(h_star)
    # print(calculate_mse(set_t[:, 0], set_t[:, 1], H[h_star]))


def calculate_mse(x_set, labels, h):
    """

    :param x_set:
    :param labels:
    :param h:
    :return:
    """
    sum = 0
    for index, x in enumerate(x_set):
        sum += (labels[index] - (np.polyval(h, x))) ** 2

    temp = (1 / len(x_set)) * sum

    return temp


def f(x):
    """

    :return:
    """
    return (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)


def y(x, sigma):
    """

    :param x:
    :param sigma:
    :return:
    """
    epsilon = np.random.normal(scale=sigma)

    return f(x) + epsilon


if __name__ == '__main__':

    X = np.random.uniform(-3.2, 2.2, 1500)

    # sigma = 1
    tags = np.array([[x, y(x, 1)] for x in X])
    D_x, T_x, D_y, T_y = train_test_split(tags[:, 0], tags[:, 1],
                                          test_size=500)

    D = np.stack((D_x, D_y), axis=1)
    T = np.stack((T_x, T_y), axis=1)

    question_10_b_c(D, T, 1)
    question_10_d(D, T, 5, 1)

    # sigma = 5

    tags_2 = np.array([[x, y(x, 5)] for x in X])
    D_x, T_x, D_y, T_y = train_test_split(tags_2[:, 0], tags_2[:, 1],
                                          test_size=500)

    D = np.stack((D_x, D_y), axis=1)
    T = np.stack((T_x, T_y), axis=1)

    question_10_b_c(D, T, 5)
    question_10_d(D, T, 5, 5)

