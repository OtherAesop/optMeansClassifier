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


def map_classes(data_set, label_set):  # takes a datasets and a corresponding labelset and summarizes
    data_sum = dict()
    if len(data_set) != len(label_set):  # Shoot warning if user passes mismatching elements
        print("ERROR: Mismatching label and attribute sets")

    for _ in range(len(data_set)):  # Iterate through list and remove and summarize data
        if label_set[0] not in data_sum.keys():  # Key does not exist, make new list
            x = label_set.pop(0)  # doing it this way guarantees everything is added into the dictionary as a list...
            data_sum[x] = []      # ...and keeps the summary neat and uniform
            data_sum[x].append(data_set.pop(0))
        else:  # Key does exist, append to end of list
            data_sum[label_set.pop(0)].append(data_set.pop(0))
    return data_sum
