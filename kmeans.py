import utility as helper
import random as rand
import math
import numpy as np
import itertools as it
from PIL import Image


class Point(object):
    def __init__(self, l, f):
        self.label = l
        self.features = f


class Cluster(object):
    def __init__(self):
        self.center = []
        self.points = []


def dist(x_vec, y_vec):  # Euclidean distance, takes and returns list
    return math.sqrt(sum([(float(a) - float(b)) ** 2 for a, b in zip(x_vec, y_vec)]))


def mse(x_vec, y_vec):
    return np.square([(float(a) - float(b)) for a, b in zip(x_vec, y_vec)]).mean()


def average_mse(cluster):
    accumulator = 0
    for point in cluster.points:
        accumulator += mse(point.features, cluster.center)
    return accumulator


def mss(cluster_centers, k):  # I would be very impressed if this worked
    accumulator = 0
    for pair in it.permutations(cluster_centers, 2):  # Sum(d(u1,u2)^2)
        accumulator += dist(pair[0], pair[1])
    return accumulator / ((k*(k-1))/2)


def entropy(cluster):
    m = len(cluster.points)
    label_dict = dict()
    for p in cluster.points:  # count m(i)
        if p.label not in label_dict:
            label_dict[p.label] = 1
        else:
            label_dict[p.label] += 1
    accumulator = 0
    for class_label in label_dict:  # mi/m * entropy(ci)
        mi = label_dict[class_label]
        accumulator += (mi / m) * math.log2(mi / m)
    return -accumulator  # entropy(ci)


# Calculates euclidean distance and updates cluster membership
def update_clusters(clusters):
    centers = list()
    changed = False
    for cluster in clusters:  # need cluster centers
        centers.append(cluster.center)
    # Find Euclidean distance for all points and update membership
    for (cl_from_dex, cluster) in enumerate(clusters):
        for (from_dex, point) in enumerate(cluster.points):
            euc_distance = list()
            for center in centers:
                euc_distance.append(dist(point.features, center))
            mindex = euc_distance.index(min(euc_distance))
            if cl_from_dex != mindex:  # the source cluster != best cluster
                clusters[mindex].points.append(cluster.points.pop(from_dex))
                changed = True
    # Calculate cluster center
    for cluster in clusters:
        m = len(cluster.points)
        accumulator = [0] * m  # Must be the same size as the feature vector
        for point in cluster.points:
            accumulator = [(float(a) + float(b)) for a, b in zip(accumulator, point.features)]
        cluster.center = [float(x) / float(m) for x in accumulator]

    return changed


# Compute MSE, MSS, and Mean Entropy
# Returns values in an ordered tuple
def calc_kmeans_stats(clusters):
    mse_accumulator = 0
    entropy_accumulator = 0
    m = 0
    for cluster in clusters:  # count total number of instances
        m += len(cluster.points)
    for cluster in clusters:
        mi = len(cluster.points)
        mse_accumulator += average_mse(cluster)
        entropy_accumulator += ((mi / m) * entropy(cluster))
    mse_accumulator = mse_accumulator / len(clusters)

    cluster_centers = [cluster.center for cluster in clusters]
    mss_value = mss(cluster_centers, len(clusters))
    return mse_accumulator, mss_value, entropy_accumulator


# Does a run of K-means training
def k_train(clusters, dataset, labelset, verbose=False):
    # Create k initial clusters and assign all points to them randomly from training data
    for (label_index, feature) in enumerate(dataset):
        point = Point(labelset[label_index], feature)
        cluster_index = rand.randrange(len(clusters))
        clusters[cluster_index].points.append(point)
    # Assign cluster center randomly from training samples
    for cluster in clusters:
        cluster.center = cluster.points[rand.randrange(len(cluster.points))].features
    # Iterate the below until convergence
    mse_, mss_, ent_ = calc_kmeans_stats(clusters)
    if verbose:
        helper.print_stats(mse_, mss_, ent_)

    counter = 0
    mse_t, mss_t, ent_t = 0, 0, 0
    converged = True
    while update_clusters(clusters):
        prev_mse, prev_mss, prev_ent = mse_t, mss_t, ent_t
        mse_, mss_, ent_ = calc_kmeans_stats(clusters)
        mse_t, mss_t, ent_t = mse_, mss_, ent_
        if verbose:
            helper.print_stats(mse_, mss_, ent_)
        counter += 1
        if (prev_mse - mse_t) + (prev_mss - mss_t) + (prev_ent - ent_t) < 5 and counter >= 100 and counter != 1:  # Probably indicates non-convergence
            converged = False
            if verbose:
                print("Non-convergence detected, finishing training sequence...")
            break

    return mse_, mss_, ent_, clusters, converged


# Returns Mean Squared Error, Mean Square Separation, and Mean Entropy in an ordered tuple
def kmeans(training_file, test_file, k=10, verbose=False, save=False):
    dataset, labelset = helper.make_datasets(training_file)  # Do input step
    t_data, t_labels = helper.make_datasets(test_file)
    no_converge = 0
    c_matrix = [[0] * (k+2) for _ in range(k)]  # 2d array of size k^2, with 2 extra columns for formatting
    for x in range(k):  # Add 'actual' label column
        c_matrix[x][k] = x
    results = list()
    mse_results = list()
    if verbose:
        print("Starting first run...")
    for counter in range(5):
        seed = rand.randint(0, 1000)
        rand.seed(seed)  # init by passed seed
        clusters = [Cluster() for _ in range(k)]  # create clusters
        mse_, mss_, ent_, clusters, converge = k_train(clusters, dataset, labelset, verbose)  # cluster until convergence
        mse_results.append(mse_)  # log results
        results.append([mss_, ent_, seed])  # log results
        if not converge:  # log non convergence
            no_converge += 1
        if verbose:
            print(f"Final mean squared error: {mse_}, starting run #{counter+1} of 5...")
    bestdex = mse_results.index(min(mse_results))  # find the index of the best run
    x, y, z = results[bestdex]
    best_run = [mse_results[bestdex], x, y, z]  # find the best run

    # Train new 'best' clustering
    rand.seed(best_run[3])  # init by passed seed
    clusters = [Cluster() for _ in range(k)]  # create clusters
    mse_, mss_, ent_, clusters, converge = k_train(clusters, dataset, labelset, verbose)  # cluster until convergence
    if not converge:  # log non convergence
        no_converge += 1

    # Associate clusters with most frequent class label
    cluster_labels = list()
    for cluster in clusters:
        label_dict = dict()
        for p in cluster.points:
            if p.label not in label_dict:
                label_dict[p.label] = 1
            else:
                label_dict[p.label] += 1
        bigval = max(val for key, val in label_dict.items())
        choice_labels = list()
        for key, val in label_dict.items():  # Find all key-val pairs that are the largest value from the original dict
            if val == bigval:
                choice_labels.append(key)
        biggest_key = rand.choice(choice_labels)
        cluster_labels.append(biggest_key)

    # Classify data to closest cluster
    centers = list()
    acc_pred = 0
    for cluster in clusters:  # need cluster centers
        centers.append(cluster.center)
    for features, label in zip(t_data, t_labels):
        euc_distance = list()
        for center in centers:
            euc_distance.append(dist(features, center))
        mindex = euc_distance.index(min(euc_distance))
        prediction = int(cluster_labels[mindex])  # Make confusion matrix
        actual = int(label)
        c_matrix[actual-1][prediction-1] += 1
        if prediction == actual:
            acc_pred += 1

    for y in range(k):  # Calculate accuracy percentages
        acc = 0
        total = 0
        for x in range(k):
            if x == y:
                acc = c_matrix[y][x]
            total += c_matrix[y][x]
        if total != 0:  # It is possible to only access indexes with a 0 value and we cannot divide by zero
            c_matrix[y][k+1] = (acc / total) * 100  # Linter might throw warning about the second index being a float
        elif total == 0 and acc == 0:  # Counts edge case where there are no instances of a class in the set
            c_matrix[y][k+1] = 100  # Linter might throw warning about the second index being a float here
        else:
            c_matrix[y][k+1] = 0  # Linter might throw warning about the second index being a float here

    accum = 0  # Counts class edge case where there are no instances of a class in the set
    for y in range(k):
        accum += c_matrix[y][k+1]
    total_acc_perc = accum / k

    result_matrix = helper.print_results_matrix(c_matrix, total_acc_perc, verbose, k)
    cluster_visuals = helper.print_cluster_centers(clusters, verbose)

    if save:
        for x, visual in enumerate(cluster_visuals):  # Saves images in test folder
            image = Image.fromarray(visual)
            image.convert('P').save("test_results/cluster_img" + str(x) + "-" + str(cluster_labels[x]) + "k" + str(k) + ".png", "PNG")

    if verbose:
        print(f"Average MSE: {best_run[0]:.2f}, MSS: {best_run[1]:.2f}, Mean Entropy: {best_run[2]:.2f}, Seed Num: {best_run[3]:.2f}")
        print(result_matrix)

    return best_run[0], best_run[1], best_run[2], best_run[3], result_matrix, no_converge
# EOF
