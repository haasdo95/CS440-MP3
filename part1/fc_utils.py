from part1.deserializer import read_labeled_data_files

def featurize(img: list, height: int, width: int, is_disjoint: bool, dimension=(28, 28)):
    """
    :param img: an 28 * 28 img
    :param height: height of the window
    :param width: width of the window
    :param is_disjoint: true if the windows don't overlap
    :return: an iterator of features
    """
    row, column = dimension
    row_stride = 1
    col_stride = 1
    if is_disjoint:
        row_stride = height
        col_stride = width
    row_bound = row - height + 1
    col_bound = column - width + 1
    if is_disjoint:
        row_bound = row - (row % height)
        col_bound = column - (column % width)

    for row_start in range(0, row_bound, row_stride):
        for col_start in range(0, col_bound, col_stride):
            feature = []
            for row_idx in range(row_start, row_start + height):
                for col_idx in range(col_start, col_start + width):
                    feature.append(img[row_idx][col_idx])
            yield tuple(feature)


def test_featurize():
    test_img, _ = next(read_labeled_data_files())
    features = featurize(test_img, 2, 3, False)
    features = list(features)
    print(len(features))


def main():
    test_featurize()


if __name__ == '__main__':
    main()