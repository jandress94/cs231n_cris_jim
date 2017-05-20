import pandas as pd
from collections import Counter
import random
import matplotlib.pyplot as plt
import operator


def generate_val_dataset(label_file = 'D:/cs231n_data/train_v2.csv', val_proportion = 0.1):
    train_labels = pd.read_csv(label_file)
    # train_labels = pd.read_csv('/mnt/d/cs231n_data/train_v2.csv')

    train_label_counter = Counter()
    label_to_file_dict = {}
    file_to_tags_dict = {}

    for index, row in train_labels.iterrows():
        labels = row['tags'].split()
        train_label_counter.update(labels)
        file_to_tags_dict[row['image_name']] = labels

        for label in labels:
            if label not in label_to_file_dict:
                label_to_file_dict[label] = []
            label_to_file_dict[label].append(row['image_name'])

    val_data = set()
    val_label_counter = Counter()

    for label in label_to_file_dict:
        for image_name in label_to_file_dict[label]:
            if random.random() < 1.0 - val_label_counter[label] / (val_proportion * train_label_counter[label]):
                val_data.add(image_name)
                val_label_counter.update(file_to_tags_dict[image_name])

    return val_data


if __name__ == '__main__':

    label_file = 'D:/cs231n_data/train_v2.csv'
    train_labels = pd.read_csv(label_file)

    val_data = generate_val_dataset(label_file = label_file)

    val_label_counter = Counter()
    train_labels = pd.read_csv(label_file)
    for index, row in train_labels.iterrows():
        labels = row['tags'].split()
        if row['image_name'] in val_data:
            val_label_counter.update(labels)

    print val_label_counter
    sorted_labels = sorted(val_label_counter.items(), key=operator.itemgetter(1))

    plt.bar(range(len(sorted_labels)), [x[1] for x in sorted_labels], align='center')
    plt.xticks(range(len(sorted_labels)), [x[0] for x in sorted_labels], rotation=-90)
    plt.ylabel('Number of Images')
    plt.xlabel('Label')
    plt.show()

