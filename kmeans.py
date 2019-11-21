import utility as helper
import random as rand


class Point(object):
    def __init__(self, l, f):
        self.label = l
        self.features = f


class Cluster(object):
    def __init__(self):
        self.center = []
        self.points = []


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
