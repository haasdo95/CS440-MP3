from part1.fc_utils import featurize
from part1.deserializer import read_labeled_data_files # for four credit
from part1.deserializer import train_cv_split # for four credit
# from part1.deserializer_face import read_labeled_data_files # for ec
import pickle

def get_num_features(kernel_size, is_disjoint):
    height, width = kernel_size
    img, _ = (read_labeled_data_files(is_training=True, is_binary=True))[0]
    features = featurize(img, height, width, is_disjoint)
    return len(list(features))


def count_occurrence(kernel_size: tuple, is_disjoint: bool, data: list):
    height, width = kernel_size
    num_features = get_num_features(kernel_size, is_disjoint)
    occurrence = tuple([[{} for _ in range(num_features)] for _ in range(10)])
    # data = read_labeled_data_files(is_training=True, is_binary=True)
    for img, label in data:
        entry = occurrence[label]
        features = featurize(img, height, width, is_disjoint)
        for idx, feature in enumerate(features):
            feature_entry = entry[idx]
            feature_entry[feature] = 1 if feature not in feature_entry else feature_entry[feature] + 1
    return occurrence

def test_count_occ():
    occ = count_occurrence((2, 2), True, [])
    print(occ)
    assert len(occ) == 10
    assert len(occ[0]) == 196

def freq2prob(occurrence: tuple, smoother: int):
    """
    this function ALTERS occurrence
    save denominator & smoother
    postpone the division
    """
    for occ_of_class in occurrence:
        for feature_count in occ_of_class:
            kernel_size = len(list(feature_count.keys())[0])
            denominator = sum(feature_count.values()) + (2 ** kernel_size) * smoother
            feature_count["denominator"] = denominator
            feature_count["smoother"] = smoother

def epoch(epoch_num: int, num_folds: int, sizes_to_run_disj, sizes_to_run_overlap, smoother, isCV=True):
    data = read_labeled_data_files()
    if isCV:
        train, _ = train_cv_split(data, epoch_num, num_folds)
        data = train
    for kernel_size in sizes_to_run_disj:
        print("WORKING ON DISJ: ", kernel_size)
        occ = count_occurrence(kernel_size, True, data)
        print(occ)
        freq2prob(occ, smoother=smoother)
        old_name = 'train_results/result_disj' +  str(kernel_size[0]) + str(kernel_size[1]) + '.pkl'
        cv_name = 'train_results/result_disjFC' + str(smoother) + "(" + str(epoch_num) + "," + str(num_folds) + ")" + str(kernel_size[0]) + str(kernel_size[1]) + '.pkl'
        name = None
        if isCV:
            name = cv_name
        else:
            name = old_name
        pickle.dump(occ, open(name, 'wb'))
    for kernel_size in sizes_to_run_overlap:
        print("WORKING ON OVLP: ", kernel_size)
        occ = count_occurrence(kernel_size, False, data)
        freq2prob(occ, smoother=smoother)
        old_name = 'train_results/result_overlap' +  str(kernel_size[0]) + str(kernel_size[1]) + '.pkl'
        cv_name = 'train_results/result_overlapFC' + str(smoother) + "(" + str(epoch_num) + "," + str(num_folds) + ")" + str(kernel_size[0]) + str(kernel_size[1]) + '.pkl'
        name = None
        if isCV:
            name = cv_name
        else:
            name = old_name
        pickle.dump(occ, open(name, 'wb'))

if __name__ == '__main__':
    num_folds = 10
    smoothers = [0.1, 0.5, 1, 2, 4, 8]
    sizes_to_run_disj = [(1, 1), (2, 2), (2, 4), (4, 2), (4, 4)]
    # sizes_to_run_disj = [(1, 1), (2, 2)]
    sizes_to_run_overlap = [(2, 2), (2, 4), (4, 2), (4, 4), (2, 3), (3, 2), (3, 3)]
    # sizes_to_run_overlap = [(2, 2)]
    for smoother in smoothers:
        for i in range(num_folds):
            epoch(i, num_folds, sizes_to_run_disj, sizes_to_run_overlap, smoother)