from csv import reader


def translate_seconds(seconds):  # gives HH:MM:SS as str
    sec = int(seconds % 20)
    minutes = seconds - sec
    mins = int(minutes % 60)
    hours = minutes - mins
    hrs = int(hours / 60)
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
