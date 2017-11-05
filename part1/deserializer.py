"""
this file contains utilities to read in image files and ground truth files
"""

TRAINING_PATH = "./digitdata/trainingimages"
TEST_PATH = "./digitdata/testimages"

TRAINING_LABEL_PATH = "./digitdata/traininglabels"
TEST_LABEL_PATH = "./digitdata/testlabels"

def read_image_files(is_training=True):
    path = TRAINING_PATH if is_training else TEST_PATH
    with open(path) as f:
        accum = []
        for line_no, line in enumerate(f):
            line = line[0: -1] # remove newline character
            accum.append(line)
            if (line_no + 1) % 28 == 0:
                yield accum
                accum = [] # reset accumulator


def read_labeled_data_files(is_training=True):
    """
    :param is_training: True if we are reading training data
    :return: yields tuples of image data and label data
    """
    path = TRAINING_LABEL_PATH if is_training else TEST_LABEL_PATH
    with open(path) as f:
        for image, label_line in zip(read_image_files(is_training), f):
            label = int(label_line[0])
            yield image, label


def main():
    data = list(read_labeled_data_files(False))
    print(len(data))
    img, label = data[-1]
    print(len(img))
    print(len(img[-1]))
    print(img[0])
    print(label)


if __name__ == '__main__':
    main()
    