import utility as helper
import random as rand
import math
import numpy as np
import itertools as it


class Point(object):
    def __init__(self, l, f):
        self.label = l
        self.features = f


class Cluster(object):
    def __init__(self):
        self.center = []
        self.points = []


# Possibly need root to match mss function?
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
        # print(accumulator)
        cluster.center = [float(x) / float(m) for x in accumulator]
        # print(cluster.center)
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


# Returns Mean Squared Error, Mean Square Separation, and Mean Entropy in an ordered tuple
def kmeans(training_file, test_file, k=10, seed=1, verbose=False):
    dataset, labelset = helper.make_datasets(training_file)  # Do input step
    rand.seed(seed)  # init by passed seed
    # Create k initial clusters and assign all points to them randomly from training data
    clusters = [Cluster() for _ in range(k)]
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
        # helper.print_clusters(clusters)
        helper.print_stats(mse_, mss_, ent_)

    while update_clusters(clusters):
        mse_, mss_, ent_ = calc_kmeans_stats(clusters)
        if verbose:
            helper.print_stats(mse_, mss_, ent_)
    
    return mse_, mss_, ent_
# EOF
