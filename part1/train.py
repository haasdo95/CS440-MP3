"""
' ' means a white (background) pixel, '+' means a gray pixel, and '#' means a black (foreground) pixel.
"""
import pickle
from part1.deserializer import read_labeled_data_files

def group_by_label(training_tuples):
    """
    :param training_tuples: iterable of (image, label) tuples
    :return: a 10-D tuple (imgs0, ..., imgs9)
    """
    grouped = tuple([[] for _ in range(10)])
    for img, label in training_tuples:
        grouped[label].append(img)
    return grouped


def count_occurrence(grouped: tuple):
    """
    :param grouped: the tuple returned by group_by_label
    :return: a tuple representing the # of occurrence of each color in each pixel of each group
    """
    # well. a case of code smell. who cares.
    groups = tuple([[[{' ': 0, '#': 0, '+': 0} for _ in range(28)] for _ in range(28)] for _ in range(10)])
    for label, imgs in enumerate(grouped):
        curr_img_cnt = groups[label]
        for img in imgs:
            for row_no, row in enumerate(img):
                for col_no, color in enumerate(row):
                    curr_img_cnt[row_no][col_no][color] += 1
    return groups


def freq2prob(groups: tuple, smoother: int):
    """
    :param groups: tuple returned by count_occurrence
    :param smoother: related to Laplace Smoothing
    :return: this function ALTERS the "groups" passed in!
    """
    for matrix_of_dict in groups:
        for row in matrix_of_dict:
            for d in row:
                denominator = sum(d.values()) + 3 * smoother
                for k in d:
                    d[k] = (d[k] + smoother) / denominator


def main():
    grouped = group_by_label(read_labeled_data_files())
    print(len(grouped))
    for g in grouped:
        print(len(g))
    occ = count_occurrence(grouped)
    assert len(occ) == 10
    assert len(occ[0]) == 28
    assert len(occ[0][0]) == 28
    assert len(occ[0][0][0].keys()) == 3 # example usage
    freq2prob(occ, 1)
    print(occ)
    pickle.dump(occ, open("train_result.pkl", "wb"))

if __name__ == '__main__':
    main()