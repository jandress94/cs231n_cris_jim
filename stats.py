import argparse
import sys

import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt 

parser = argparse.ArgumentParser()

parser.add_argument('--label_list_file', default = '../cs231n_data/labels.txt')
parser.add_argument('--val', default='../cs231n_data/val.csv')
parser.add_argument('--train', default='../cs231n_data/train_v2.csv')
parser.add_argument('--loss', default='../cs231n_data/loss.txt')
parser.add_argument('--root', default='../cs231n_data/')
parser.add_argument('--log', default='../cs231n_data/f2.txt')

def find_classes(label_list_file):
    f = open(label_list_file)
    classes = np.array([line.strip() for line in f])
    f.close()
    return classes

def precision_and_recall(args):
    classes = find_classes(args.label_list_file)
    num_classes = len(classes)

    class_to_num = {}
    for i in range(num_classes):
        class_to_num[classes[i]] = i

    val_dict = {}
    train_dict = {}

    i = 0
    with open(args.train) as f:
        for line in f:
            if i != 0:
                split_line = line.split(',')
                file_name = split_line[0]
                file_classes = split_line[1].split(' ')
                file_classes = [file_class.strip() for file_class in file_classes]
                train_dict[file_name] = file_classes
            i += 1

    i = 0
    with open(args.val) as f:
        for line in f:
            if i != 0:
                split_line = line.split(',')
                file_name = split_line[0]
                file_classes = split_line[1].split(' ')
                file_classes = [file_class.strip() for file_class in file_classes]
                val_dict[file_name] = file_classes
            i += 1

    counts = np.zeros((3, num_classes))
    for file_name in val_dict.keys():
        preds = set(val_dict[file_name])
        y = set(train_dict[file_name])
        common = preds & y 
        for c in common:
            counts[0, class_to_num[c]] += 1
        for c in preds:
            counts[1, class_to_num[c]] += 1
        for c in y:
            counts[2, class_to_num[c]] += 1

    
    with open('stats.csv', 'w') as csvfile:
        fieldnames = ['Class', 'Precision', 'Recall']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for c in range(num_classes):
            writer.writerow({'Class': classes[c], 'Precision': counts[0, c] / counts[1, c], 'Recall': counts[0, c] / counts[2, c]})

def plot_loss_and_f2(args):
    train_f2 = []
    val_f2 = []
    epoch_f2 = range(1, 31)

    with open(args.log) as f:
        for line in f:
            if 'Val f2:' in line:
                split_line = line.split('  ')
                val_f2.append(float(split_line[1].strip()))
            if 'Train f2:' in line:
                split_line = line.split('  ')
                train_f2.append(float(split_line[1].strip()))

    print len(val_f2)
    print len(train_f2)
    loss = []
    epoch = []
    with open(args.loss) as f:
        i = 0
        avg = []
        for line in f:
            if i < 5: # there are 575 printed loss values per epoch
                loss.append(float(line.strip()))
                epoch.append(i / 575.0)
            else:
                if i % 10 == 5:
                    loss.append(np.mean(avg))
                    epoch.append(i/575.0)
                    avg = []
                avg.append(float(line.strip()))

            i += 1

    #plt.scatter(epoch, loss)
    #plt.show()

    fig, ax1 = plt.subplots(figsize = (5, 5))
    ax1.plot(epoch, loss, label = 'Training Loss')
    ax1.set_xlabel('Epoch', size = 25)
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Loss', size =25)
    #ax1.tick_params('y', colors='b')
    ax1.tick_params(labelsize=25)
    ax1.legend(loc='lower right', prop={'size':20})

    ax2 = ax1.twinx()
    ax2.plot(epoch_f2, val_f2, 'g', label = 'Validation F2')
    ax2.plot(epoch_f2, train_f2, 'r', label = 'Training F2')
    ax2.set_ylabel('F2 Score', size = 25)
    ax2.tick_params(labelsize=25)
    ax2.legend(loc = 'center right', prop={'size':20})
    
    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    args = parser.parse_args()
    #precision_and_recall(args)
    plot_loss_and_f2(args)


