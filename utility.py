from csv import reader
import texttable as ttable
import numpy as np


def translate_seconds(seconds):  # gives HH:MM:SS as str
    sec = int(seconds % 60)
    mins = int(seconds / 60 % 60)
    hrs = int(seconds / 60 / 60)
    return str(hrs).zfill(2) + ":" + str(mins).zfill(2) + ":" + str(sec).zfill(2)


def make_datasets(filepath):  # Separates labels and preps data sets
    data_set = []
    label_set = []
    with open(filepath) as data_file:
        raw_data = reader(data_file, delimiter=',')  # Read in all data, includes empty spaces
        for row in raw_data:
            row[:] = (val for val in row if val != '')  # removes all empty elements created by reader
            label_set.append(row.pop())  # remove last element which is label and put in labelset
            data_set.append(row)
    data_file.close()
    return data_set, label_set


def print_clusters(clusters):
    print("Cluster length: ", len(clusters))
    for c_index, cluster in enumerate(clusters):
        print("Cluster ", c_index+1, "Points: ", len(cluster.points))


def print_stats(mse, mss, entropy):
    print(f"Mean Squared Error: {mse:.2f} Mean Square Separation: {mss:.2f} Mean Entropy: {entropy:.2f}")


def print_results_matrix(matrix, total_acc, verbose, k):

    c_arr = ["c"] * (k+2)
    nine_arr = ["9"] * k
    dash_arr = ["-"] * k
    num_arr = list()
    for x in range(k):
        num_arr.append(x)
    num_arr.append("Predicted x Actual")
    num_arr.append("Accuracy")
    nine_arr.append("14")
    nine_arr.append("12")
    dash_arr.append("Total Avg Acc")
    dash_arr.append(total_acc)

    table1 = ttable.Texttable().set_cols_align(c_arr) \
        .set_cols_valign(c_arr) \
        .set_cols_width(nine_arr) \
        .add_row(num_arr) \
        .add_rows(matrix, False) \
        .add_row(dash_arr)
    table_str = table1.draw() + "\n"

    if verbose:
        print(table_str)

    return table_str


# Prepares cluster visualization and prints to console if verbose is true
def print_cluster_centers(clusters, verbose=False):
    cluster_visuals = list()
    for _ in range(len(clusters)):  # make an 8x8 grid for each cluster
        arr = np.asarray([np.asarray([0 for _ in range(8)]) for _ in range(8)])
        if arr[0] is arr[1]:
            print(True)
        cluster_visuals.append(arr)

    for cdex, cluster in enumerate(clusters):
        visual = cluster_visuals[cdex]
        copy_cluster = [x for x in cluster.center]
        maxval = max(copy_cluster)
        minval = min(copy_cluster)
        for y, column in enumerate(visual):
            for x in range(len(column)):
                if not copy_cluster:  # cluster may have no members and thus no center
                    if verbose:
                        print("Incomplete cluster visual detected at (y,x) index: ", len(visual) - 1 - y, len(column) - 1 - x)
                    continue
                visual[len(visual) - 1 - y][len(column) - 1 - x] = round((copy_cluster.pop() - minval) / (maxval - minval) * 255)

    if verbose:
        for cnum, visual in enumerate(cluster_visuals):
            print("Cluster #", cnum+1)
            for row in visual:
                print(row)
    return cluster_visuals


def save(time, k, mse, mss, ent, seed, cmatrix, non_convergence_inst):  # Saves important test info
    log = open(f"test_results/kmeans_k{k}.txt", "w")
    log.write(f"{k} clusters, {seed} seed, {time} runtime total, {non_convergence_inst} instances of non-convergence.\n")
    log.write("Format is as follows {Average Mean Square Error, Mean Square Separation, Mean Entropy}\n")
    log.write(f"Average MSE: {mse}, MSS: {mss}, ENT: {ent}\n")
    log.write(f'Confusion Matrix for tests\n')
    log.write(cmatrix)
    log.close()

