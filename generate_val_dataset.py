import pandas as pd
from collections import Counter
import random
import matplotlib.pyplot as plt
import operator
from shutil import copyfile

def generate_val_dataset(label_file, val_proportion = 0.1):
    train_labels = pd.read_csv(label_file)

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

    all_label_filename = '../cs231n_data/train_v2-all.csv'
    train_label_filename = '../cs231n_data/train_v2.csv'
    val_label_filename = '../cs231n_data/val_v2.csv'
    all_data_dir = '../cs231n_data/train-jpg-all/'
    train_dir = '../cs231n_data/train-jpg/'
    val_dir = '../cs231n_data/val-jpg/'

    train_labels = pd.read_csv(all_label_filename)

    val_data = generate_val_dataset(label_file = all_label_filename)

    train_label_file = open(train_label_filename, 'w')
    train_label_file.write('image_name,tags')
    val_label_file = open(val_label_filename, 'w')
    val_label_file.write('image_name,tags')

    val_label_counter = Counter()
    for index, row in train_labels.iterrows():
        labels = row['tags'].split()
        if row['image_name'] in val_data:
            val_label_counter.update(labels)

            val_label_file.write('\n%s,%s' % (row['image_name'], row['tags']))
            copyfile('%s%s.jpg' % (all_data_dir, row['image_name']), '%s%s.jpg' % (val_dir, row['image_name']))
        else:
            train_label_file.write('\n%s,%s' % (row['image_name'], row['tags']))
            copyfile('%s%s.jpg' % (all_data_dir, row['image_name']), '%s%s.jpg' % (train_dir, row['image_name']))

    train_label_file.flush()
    train_label_file.close()
    val_label_file.flush()
    val_label_file.close()

    print(val_label_counter)
    sorted_labels = sorted(val_label_counter.items(), key=operator.itemgetter(1))
