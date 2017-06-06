import argparse
import sys

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument('--label_list_file', default = '../cs231n_data/labels.txt')
# Assume best model is the first in list
parser.add_argument('--submission_list_file', default='../cs231n_data/submissions.txt')

parser.add_argument('--root', default='../cs231n_data/')

parser.add_argument('--save_path', default='../cs231n_data/ensemble.csv')

parser.add_argument('--best_multiplier', default=1, type=int)

parser.add_argument('--threshold', default=0.5, type=float)

def find_classes(label_list_file):
    f = open(label_list_file)
    classes = np.array([line.strip() for line in f])
    f.close()
    return classes

def main(args):
    csv_files = []
    num_models = 0
    with open(args.submission_list_file) as f:
    	for csv_file in f:
    		if num_models == 0:
    			for i in range(args.best_multiplier):
    				csv_files.append(args.root + csv_file.strip())
    		else:
    			csv_files.append(args.root + csv_file.strip())
    		num_models += 1
    print('Creating an ensemble for the files ' + str(csv_files))
    classes = find_classes(args.label_list_file)
    num_classes = len(classes)

    num_models = len(csv_files)
    for i in range(num_models):
        df = pd.read_csv(csv_files[i])
        for c in classes:
            df[c] = df['tags'].apply(lambda x: 1 if c in x.split(' ') else 0)

        if i == 0:
            names  = df.iloc[:,0].values
            N = df.shape[0]
            predictions = np.zeros((N, num_classes), dtype = np.float32)

        table = df.iloc[:,2:].values.astype(np.float32)
        predictions = predictions + table

    predictions = predictions / num_models
    predictions = predictions > args.threshold
    predictions = predictions.astype(np.int)
    predictions = [' '.join(classes[y_pred_row == 1]) for y_pred_row in predictions]

    subm = pd.DataFrame()
    subm['image_name'] = names
    subm['tags'] = predictions
    subm.to_csv(args.save_path, index=False)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

