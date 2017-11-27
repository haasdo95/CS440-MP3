from part1.test_fc import measure_performance_CV, print_confusion_matrix
from part1.deserializer import train_cv_split, read_labeled_data_files
from part1.train_fc import epoch

def getmax(lst):
    m = -1
    ci = 0
    for i, elem in enumerate(lst):
        if elem > m:
            m = elem
            ci = i
    return ci

if __name__ == '__main__':
    smoothers = [0.1, 0.5, 1, 2, 4, 8]
    sizes_to_run_disj = [(1, 1), (2, 2), (2, 4), (4, 2), (4, 4)]
    sizes_to_run_overlap = [(2, 2), (2, 4), (4, 2), (4, 4), (2, 3), (3, 2), (3, 3)]
    num_folds = 10
    disj_rec = {}
    ovlp_rec = {}
    for kernel_size in sizes_to_run_disj:
        smoother_perfs = []
        for smoother in smoothers:
            print("USING SMOOTHER: ", smoother)
            fold_perfs = []
            for fold_idx in range(num_folds):
                test_data = read_labeled_data_files(is_training=True, is_binary=True)
                _, test_data = train_cv_split(test_data, fold_idx, num_folds)
                perf, conf_matrix = measure_performance_CV(test_data, kernel_size, True, smoother, fold_idx, num_folds)
                fold_perfs.append(perf)
                print("DISJ PERFORMANCE ON ", str(kernel_size), ": ", perf)
                # print_confusion_matrix(conf_matrix)
            smoother_perfs.append((sum(fold_perfs) / len(fold_perfs)))
        disj_rec[kernel_size] = smoothers[getmax(smoother_perfs)]

    for kernel_size in sizes_to_run_overlap:
        smoother_perfs = []
        for smoother in smoothers:
            print("USING SMOOTHER: ", smoother)
            fold_perfs = []
            for fold_idx in range(num_folds):
                test_data = read_labeled_data_files(is_training=True, is_binary=True)
                _, test_data = train_cv_split(test_data, fold_idx, num_folds)
                perf, conf_matrix = measure_performance_CV(test_data, kernel_size, False, smoother, fold_idx, num_folds)
                fold_perfs.append(perf)
                print("OVLP PERFORMANCE ON ", str(kernel_size), ": ", perf)
                # print_confusion_matrix(conf_matrix)
            smoother_perfs.append((sum(fold_perfs) / len(fold_perfs)))
        ovlp_rec[kernel_size] = smoothers[getmax(smoother_perfs)]
    try:
        print(disj_rec)
        with open("disj_rec.txt", "w") as f:
            f.write(str(disj_rec))
        print(ovlp_rec)
        with open("ovlp_rec.txt", "w") as f:
            f.write(str(ovlp_rec))
    except:
        pass
    for kernel_size, best_smoother in disj_rec.items():
        epoch(-1, -1, [kernel_size], [], best_smoother, isCV=False)
    for kernel_size, best_smoother in ovlp_rec.items():
        epoch(-1, -1, [], [kernel_size], best_smoother, isCV=False)