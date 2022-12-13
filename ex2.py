import numpy as np
import argparse


def CTCLoss(network_output_path: str, input_labels: str, alphabet: str) -> float:
    # read the output matrix y
    y_probability_matrix = np.load(network_output_path)
    T, K = y_probability_matrix.shape  # extract shape from matrix

    # construct alphabet dictionary
    alphabet += "-"  # add empty character
    alphabet_dict = {alphabet[i]: i for i in range(len(alphabet))}

    # create padded sequence
    z = '-'
    for i in range(len(input_labels)):
        z += input_labels[i] + '-'
    L = len(z)

    # initialize alpha
    alpha = np.zeros((T, L))
    alpha[0, 0] = y_probability_matrix[0, -1]  # epsilon
    alpha[0, 1] = y_probability_matrix[0, alphabet_dict[z[1]]]

    # use dynamic programing to calculate alpha
    for t in range(1, T):
        for s in range(L):
            additive_term = alpha[t - 1, s]
            if s > 0:
                additive_term += alpha[t - 1, s - 1]
            if (s > 1) and (z[s] != '-') and (z[s] != z[s - 2]):
                additive_term += alpha[t - 1, s - 2]
            alpha[t, s] = additive_term * y_probability_matrix[t, alphabet_dict[z[s]]]

    return np.round(alpha[-1, -2] + alpha[-1, -1], 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("network_output_path", type=str, help="Path to the network output matrix")
    parser.add_argument("input_labels", help="input labels")
    parser.add_argument("alphabet", help="alphabet")
    args = parser.parse_args()
    result = CTCLoss(args.network_output_path, args.input_labels, args.alphabet)
    # write result to file
    with open("output.txt", "w") as f:
        f.write(str(result))
