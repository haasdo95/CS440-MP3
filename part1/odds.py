"""
this file creates the "odds" heatmap
"""

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
