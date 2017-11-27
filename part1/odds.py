"""
this file creates the "odds" heatmap
"""

import os
from visualization.viz import *

from part1.train import retrieve_prob
lookup_table = retrieve_prob(is_binary=True)

def get_odds_between(class1: int, class2: int):
    entry1 = lookup_table[class1]
    entry2 = lookup_table[class2]
    odd_matrix = [[0 for _ in range(28)] for _ in range(28)]
    for i in range(28):
        for j in range(28):
            odd_matrix[i][j] = (1 - entry1[i][j][' ']) / (1 - entry2[i][j][' '])
    return odd_matrix

def get_prob_matrix_of(number: int):
    assert number >= 0 and number < 10
    entry = lookup_table[number]
    matrix = [[0 for _ in range(28)] for _ in range(28)]
    for i in range(28):
        for j in range(28):
            matrix[i][j] = 1 - entry[i][j][' ']
    return matrix

if __name__ == '__main__':
    # example usage
    # one_mat = get_prob_matrix_of(1)
    # eight_mat = get_prob_matrix_of(8)
    # one_eight_matrix = get_odds_between(1, 8)
    # print(one_mat)
    # print(eight_mat)
    # print(one_eight_matrix)
    parent_dir = os.path.dirname(os.path.dirname(__file__))
    pairs = [(0, 9), (1, 8), (2, 5), (3, 4)]
    for i, j in pairs:
        file_name = str(parent_dir) + '/report/img/heatmap (' + str(i) + ',' + str(j) + ').png'
        i_mat = get_prob_matrix_of(i)
        j_mat = get_prob_matrix_of(j)
        i_j_mat = get_odds_between(i, j)
        plot_heat_map(i, j, i_mat, j_mat, i_j_mat, fname=file_name)
        print("HEAT MAP GENERATED: (", i, j, ")")