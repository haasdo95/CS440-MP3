"""
this file contains utilities to read in image files and ground truth files
"""
import random
random.seed(666)

TRAINING_PATH = "./facedata/facedatatrain"
TEST_PATH = "./facedata/facedatatest"

TRAINING_LABEL_PATH = "./facedata/facedatatrainlabels"
TEST_LABEL_PATH = "./facedata/facedatatestlabels"


def get_prior():
    prior = [0 for _ in range(2)]
    path = TRAINING_LABEL_PATH
    f = open(path)
    for line in f:
        label = int(line[0])
        prior[label] += 1
    for i in range(2):
        prior[i] /= 451
    print(prior)
    return prior


def read_image_files(is_training=True, is_binary=True):
    path = TRAINING_PATH if is_training else TEST_PATH
    with open(path) as f:
        accum = []
        for line_no, line in enumerate(f):
            line = line[0: -1] # remove newline character
            accum.append(line)
            if (line_no + 1) % 70 == 0:
                yield accum
                accum = [] # reset accumulator


def read_labeled_data_files_private(is_training=True, is_binary=True):
    """
    :param is_training: True if we are reading training data
    :return: yields tuples of image data and label data
    """
    path = TRAINING_LABEL_PATH if is_training else TEST_LABEL_PATH
    with open(path) as f:
        for image, label_line in zip(read_image_files(is_training, is_binary=is_binary), f):
            label = int(label_line[0])
            yield image, label

once = False
def read_labeled_data_files(is_training=True, is_binary=True):
    labeled_data = list(read_labeled_data_files_private(is_training, is_binary))
    global once
    if not once:
        once = True
        random.shuffle(labeled_data)
    return labeled_data

def train_cv_split(labeled_data: list, fold_idx, num_folds):
    assert len(labeled_data) % num_folds == 0
    fold_size = len(labeled_data) // num_folds
    fold_start = fold_idx * fold_size
    prev = labeled_data[0: fold_start]
    next = labeled_data[fold_start+fold_size:]
    train_set = prev + next
    return train_set, labeled_data[fold_start: fold_start + fold_size]

def main():
    get_prior()
    data = list(read_labeled_data_files(True))
    print(len(data))
    img, label = data[-1]
    print(len(img))
    print(len(img[-1]))
    print(img[0])
    print(label)


if __name__ == '__main__':
    main()
    