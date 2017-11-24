from part1.fc_utils import featurize
from part1.deserializer_face import read_labeled_data_files # for ec
import pickle

def get_num_features(kernel_size, is_disjoint):
    height, width = kernel_size
    img, _ = next(read_labeled_data_files(is_training=True, is_binary=True))
    features = featurize(img, height, width, is_disjoint, dimension=(70, 60))
    features = list(features)
    print("NUM FEATURE: ", len(features))
    return len(features)


def count_occurrence(kernel_size: tuple, is_disjoint: bool):
    height, width = kernel_size
    num_features = get_num_features(kernel_size, is_disjoint)
    occurrence = tuple([[{} for _ in range(num_features)] for _ in range(2)])
    data = read_labeled_data_files(is_training=True, is_binary=True)
    for img, label in data:
        entry = occurrence[label]
        features = featurize(img, height, width, is_disjoint, dimension=(70, 60))
        for idx, feature in enumerate(features):
            feature_entry = entry[idx]
            feature_entry[feature] = 1 if feature not in feature_entry else feature_entry[feature] + 1
    return occurrence

def test_count_occ():
    occ = count_occurrence((2, 2), True)
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
        for feature_count in (occ_of_class):
            kernel_size = len(list(feature_count.keys())[0])
            denominator = sum(feature_count.values()) + (2 ** kernel_size) * smoother
            feature_count["denominator"] = denominator
            feature_count["smoother"] = smoother

if __name__ == '__main__':
    sizes_to_run_disj = [(1, 1), (2, 2), (2, 4), (4, 2), (4, 4)]
    sizes_to_run_overlap = [(2, 2), (2, 4), (4, 2), (4, 4), (2, 3), (3, 2), (3, 3)]
    for kernel_size in sizes_to_run_disj:
        print("WORKING ON DISJ: ", kernel_size)
        occ = count_occurrence(kernel_size, is_disjoint=True)
        print("OCC: ", occ)
        freq2prob(occ, smoother=1)
        pickle.dump(occ, open('train_results/result_disj_EC' + str(kernel_size[0]) + str(kernel_size[1]) + '.pkl', 'wb'))
    for kernel_size in sizes_to_run_overlap:
        print("WORKING ON OVLP: ", kernel_size)
        occ = count_occurrence(kernel_size, is_disjoint=False)
        freq2prob(occ, smoother=1)
        pickle.dump(occ, open('train_results/result_overlap_EC' + str(kernel_size[0]) + str(kernel_size[1]) + '.pkl', 'wb'))
