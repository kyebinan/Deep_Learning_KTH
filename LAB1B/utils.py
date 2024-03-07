import numpy as np


def approximation_func(X, Y):
    return np.exp(-X**2 * 0.1) * np.exp(-Y.T**2 * 0.1) - 0.5


def approximation_func1(X, Y):
    m = np.max(X)
    n = np.max(Y)
    a = 1/(m*n)
    return a*X*Y



def main():
    print("utils module")


if __name__ == '__main__':
    main()