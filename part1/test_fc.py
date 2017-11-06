import pickle
from math import log
from part1.fc_utils import featurize
from part1.deserializer import read_labeled_data_files, get_prior

prior = get_prior()

def prob_of_image_being(img, manuscript_num: int, kernel_size, prob_lookup_table, is_disjoint):
    entry = prob_lookup_table[manuscript_num]
    features = featurize(img, kernel_size[0], kernel_size[1], is_disjoint)
    log_sum = 0
    for idx, feature in enumerate(features):
        f_entry = entry[idx]
        smoother = f_entry["smoother"]
        denom = f_entry["denominator"]
        freq = smoother if feature not in f_entry else smoother + f_entry[feature]
        prob = freq / denom
        log_sum += log(prob)
    log_sum += log(prior[manuscript_num])
    return log_sum


def guess(img, kernel_size, prob_lookup_table, is_disjoint):
    all_probs = []
    for i in range(10):
        all_probs.append(prob_of_image_being(img, i, kernel_size, prob_lookup_table, is_disjoint))
    max_index = all_probs.index(max(all_probs))
    return max_index


def measure_performance(test_data, kernel_size, is_disjoint):
    height, width = kernel_size
    post_fix = "_disj" if is_disjoint else "_overlap"
    path = 'train_results/result' + post_fix + str(height) + str(width) + '.pkl'
    prob_lookup_table = pickle.load(open(path, 'rb'))

    # bookkeeping comes here!
    test_data = list(test_data)
    total_count = len(test_data)
    overall_correctness = 0
    record = []  # store tuples of (truth, my)

    for img, truth in test_data:
        my_guess = guess(img, kernel_size, prob_lookup_table, is_disjoint)
        record.append((truth, my_guess))
        if my_guess == truth:
            overall_correctness += 1
    return (overall_correctness / total_count), analyze(record, kernel_size, is_disjoint)


def analyze(record, kernel_size, is_disjoint):
    """
    :param record: list of (truth, mine) tuples
    :return: confusion matrix etc
    """
    disj_text = "Disjoint" if is_disjoint else "Overlap"
    print("Creating Conf Matrix for: ", str(kernel_size), " ", disj_text)
    confusion_matrix = [[0 for _ in range(10)] for _ in range(10)]
    for truth, mine in record:
        confusion_matrix[truth][mine] += 1
    for row in confusion_matrix:  # converted to prob
        total = sum(row)
        for i in range(len(row)):
            row[i] = row[i] / total
    return confusion_matrix


def print_confusion_matrix(conf_mat):
    for row in conf_mat:
        row_str = ""
        for c in row:
            row_str += str(round(c, 4))
            row_str += '\t'
        print(row_str)


if __name__ == '__main__':
    sizes_to_run_disj = [(1, 1), (2, 2), (2, 4), (4, 2), (4, 4)]
    sizes_to_run_overlap = [(2, 2), (2, 4), (4, 2), (4, 4), (2, 3), (3, 2), (3, 3)]
    for kernel_size in sizes_to_run_disj:
        test_data = read_labeled_data_files(is_training=False, is_binary=True)
        perf, conf_matrix = measure_performance(test_data, kernel_size, True)
        print("DISJ PERFORMANCE ON ", str(kernel_size), ": ", perf)
        print_confusion_matrix(conf_matrix)
    for kernel_size in sizes_to_run_overlap:
        test_data = read_labeled_data_files(is_training=False, is_binary=True)
        perf, conf_matrix = measure_performance(test_data, kernel_size, False)
        print("OVLP PERFORMANCE ON ", str(kernel_size), ": ", perf)
        print_confusion_matrix(conf_matrix)