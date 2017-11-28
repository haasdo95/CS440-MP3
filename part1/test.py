from part1.deserializer import read_labeled_data_files
import pickle, os, warnings
from math import log
from pprint import pprint
from visualization.viz import *

prob = pickle.load(open("train_result.pkl", "rb"))
print(prob)


def prob_of_image_being(image, manuscript_num: int):
    """
    the probability of image being manuscript_num
    :param image: a list of strings (28 * 28)
    :return: sum of log probability
    """
    total_prob = 0
    entry = prob[manuscript_num]
    for row_no, row in enumerate(image):
        for col_no, pixel in enumerate(row):
            p = entry[row_no][col_no][pixel]
            total_prob += log(p)
    return total_prob


def get_prototypical(data):
    """
    :param data: iterable of (image, truth) tuples
    :return:
    """
    prototypical_minimax = [{
        "max": float("-inf"), "min": float("inf"),
        "max_chosen": None, "min_chosen": None
    } for _ in range(10)]
    for img, label in data:
        probability = prob_of_image_being(img, label)
        entry = prototypical_minimax[label]
        if probability < entry['min']:
            entry['min'] = probability
            entry['min_chosen'] = img
        if probability > entry['max']:
            entry['max'] = probability
            entry['max_chosen'] = img
    return prototypical_minimax

def guess(image):
    """
    :param image: a list of strings (28 * 28)
    :return the guess made on the image
    """
    all_probs = []
    for i in range(10):
        all_probs.append(prob_of_image_being(image, i))
    # get the maximum index
    max_index = all_probs.index(max(all_probs))
    return max_index


def main():
    test_data = list(read_labeled_data_files(is_training=False))
    test_data = test_data[0:1000]
    right_count = 0
    correct_for_each = [0 for _ in range(10)]
    count_each = [0 for _ in range(10)]
    # row is ground truth; col is my result
    confusion_matrix = [[0 for _ in range(10)] for _ in range(10)]
    for img, truth_label in test_data:
        my_result = guess(img)
        confusion_matrix[truth_label][my_result] += 1
        count_each[truth_label] += 1
        if my_result == truth_label:
            right_count += 1
            correct_for_each[truth_label] += 1
            print("CORRECT!: ", my_result)
        else:
            print("SHOULD BE: ", truth_label, "; I GET: ", my_result)
    # for i in range(10):
    #     correct_for_each[i] = correct_for_each[i] / count_each[i]
    #     print("CORRECTNESS FOR ", i, ": ", correct_for_each[i])
    for row in confusion_matrix:
         print(row)
    parent_dir = os.path.dirname(os.path.dirname(__file__))
    # know open issue when using savefig()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        title = 'Confusion Matrix with Disjoint Kernel Size = (1,1) and Ternary Features'
        file_name = str(parent_dir) + '/report/img/ternary.png'
        plot_confusion_matrix(cm=confusion_matrix, classes=list([str(i) for i in range(10)]),
                              fname=file_name, normalize=True, title=title)
        print("CONF MAT GENERATED: ", title)
    # print("RATE OF CORRECTNESS: ", right_count / len(test_data))
    proto = get_prototypical(test_data)
    # customized pretty print of porto
    for digit in range(len(proto)):
        proto_pp = []
        proto_pp.append('======================== Digit = ' + str(digit) + ' ========================')
        proto_pp.append('---- Highest Posterior -----' + '---' + '----- Lowest Posterior -----')
        proto_pp.append([proto[digit]['max_chosen'][idx] + ' | ' + proto[digit]['min_chosen'][idx]
                         for idx in range(len(proto[digit]['max_chosen']))])
        path_to_file = str(parent_dir) + '/report/img/prototypical/' + str(digit) + '.txt'
        with open(path_to_file, 'wt') as out:
            pprint(proto_pp, stream=out)


if __name__ == '__main__':
    main()