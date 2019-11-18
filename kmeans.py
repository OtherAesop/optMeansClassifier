import utility as helper


def kmeans(training_file, test_file, k=10, seed=1, verbose=False):
    dataset, labelset = helper.make_datasets(training_file)  # Do input step
    data_summary = helper.map_classes(dataset, labelset)  # Organizes text into key-indexed dictionary

    if verbose:
        for class_key in data_summary:
            print(class_key, len(data_summary[class_key]))

    # returns stuff as a placeholder
    return seed, k, 456
# EOF
