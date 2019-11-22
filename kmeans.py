import utility as helper
import random as rand
import math
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
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(x_vec, y_vec)]))


def mse(x_vec, y_vec, ax):
    return ((x_vec - y_vec)**2).mean(axis=ax)


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
    # 3) Find Euclidean distance for all points and update membership
    # 4) Calculate cluster center (and update)
    # 5) Return Mean Square Error, Mean-Square-Separation, and Mean Entropy (using class labels)

    if verbose:
        print("Cluster length: ", len(clusters))
        for c_index, cluster in enumerate(clusters):
            print("Cluster ", c_index+1, "Points: ", len(cluster.points), "Center: ", cluster.center)

    # returns stuff as a placeholder
    return seed, k, 456
# EOF
