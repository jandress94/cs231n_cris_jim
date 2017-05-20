import numpy as np
import os
import pandas as pd
import random
from tqdm import tqdm
import xgboost as xgb
from generate_val_dataset import generate_val_dataset

import scipy
from sklearn.metrics import fbeta_score

from PIL import Image

random_seed = 0
random.seed(random_seed)
np.random.seed(random_seed)


# Load data
train_path = '/mnt/d/cs231n_data/train-jpg/'
train_label_file = '/mnt/d/cs231n_data/train_v2.csv'
train = pd.read_csv(train_label_file)
submission_file = '/mnt/d/cs231n_data/submission.csv'

val_data = generate_val_dataset(label_file = train_label_file)

def extract_features(df, data_path, isTrain):
    im_features = pd.DataFrame()    #df.copy()

    image_names = []
    tags = []

    r_mean = []
    g_mean = []
    b_mean = []
    s_mean = []

    r_std = []
    g_std = []
    b_std = []
    s_std = []

    r_max = []
    g_max = []
    b_max = []
    s_max = []

    r_min = []
    g_min = []
    b_min = []
    s_min = []

    r_kurtosis = []
    g_kurtosis = []
    b_kurtosis = []
    s_kurtosis = []
    
    r_skewness = []
    g_skewness = []
    b_skewness = []
    s_skewness = []

    for index, row in tqdm(df.iterrows(), miniters=100):
        image_name = row['image_name']

        if isTrain == (image_name in val_data):
            continue

        image_names.append(image_name)
        tags.append(row['tags'])

        im = Image.open(data_path + image_name + '.jpg')
        im = np.array(im)[:,:,:3]
        
        r = im[:,:,0].ravel()
        g = im[:,:,1].ravel()
        b = im[:,:,2].ravel()
        s = ((255.-r)*(255.-g)*(255.-b))**(1.0 / 3.0)

        r_mean.append(np.mean(r))
        g_mean.append(np.mean(g))
        b_mean.append(np.mean(b))
        s_mean.append(np.mean(s))

        r_std.append(np.std(r))
        g_std.append(np.std(g))
        b_std.append(np.std(b))
        s_std.append(np.std(s))

        r_max.append(np.max(r))
        g_max.append(np.max(g))
        b_max.append(np.max(b))
        s_max.append(np.max(s))

        r_min.append(np.min(r))
        g_min.append(np.min(g))
        b_min.append(np.min(b))
        s_min.append(np.min(s))

        r_kurtosis.append(scipy.stats.kurtosis(r))
        g_kurtosis.append(scipy.stats.kurtosis(g))
        b_kurtosis.append(scipy.stats.kurtosis(b))
        s_kurtosis.append(scipy.stats.kurtosis(s))
        
        r_skewness.append(scipy.stats.skew(r))
        g_skewness.append(scipy.stats.skew(g))
        b_skewness.append(scipy.stats.skew(b))
        s_skewness.append(scipy.stats.skew(s))

    im_features['image_name'] = image_names
    im_features['tags'] = tags

    im_features['r_mean'] = r_mean
    im_features['g_mean'] = g_mean
    im_features['b_mean'] = b_mean
    im_features['s_mean'] = s_mean

    im_features['r_std'] = r_std
    im_features['g_std'] = g_std
    im_features['b_std'] = b_std
    im_features['s_std'] = s_std

    im_features['r_max'] = r_max
    im_features['g_max'] = g_max
    im_features['b_max'] = b_max
    im_features['s_max'] = s_max

    im_features['r_min'] = r_min
    im_features['g_min'] = g_min
    im_features['b_min'] = b_min
    im_features['s_min'] = s_min

    im_features['r_kurtosis'] = r_kurtosis
    im_features['g_kurtosis'] = g_kurtosis
    im_features['b_kurtosis'] = b_kurtosis
    im_features['s_kurtosis'] = s_kurtosis
    
    im_features['r_skewness'] = r_skewness
    im_features['g_skewness'] = g_skewness
    im_features['b_skewness'] = b_skewness
    im_features['s_skewness'] = s_skewness
    
    return im_features

# Extract features
print('Extracting train features')
train_features = extract_features(train, train_path, True)
print('Extracting test features')
val_features = extract_features(train, train_path, False)

# Prepare data
X = np.array(train_features.drop(['image_name', 'tags'], axis=1))
y_train = []

flatten = lambda l: [item for sublist in l for item in sublist]
labels = np.array(list(set(flatten([l.split(' ') for l in train_features['tags'].values]))))

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

for index, row in tqdm(train.iterrows(), miniters=100):
    tags = row['tags']

    if isTrain == (image_name in val_data):
        continue
        
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1 
    y_train.append(targets)
    
y = np.array(y_train, np.uint8)

print('X.shape = ' + str(X.shape))
print('y.shape = ' + str(y.shape))

n_classes = y.shape[1]

X_test = np.array(val_features.drop(['image_name', 'tags'], axis=1))

# Train and predict with one-vs-all strategy
y_pred = np.zeros((X_test.shape[0], n_classes))

print('Training and making predictions')
for class_i in tqdm(range(n_classes), miniters=1): 
    model = xgb.XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=100, \
                              silent=True, objective='binary:logistic', nthread=-1, \
                              gamma=0, min_child_weight=1, max_delta_step=0, \
                              subsample=1, colsample_bytree=1, colsample_bylevel=1, \
                              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, \
                              base_score=0.5, seed=random_seed, missing=None)
    model.fit(X, y[:, class_i])
    y_pred[:, class_i] = model.predict_proba(X_test)[:, 1]

preds = [' '.join(labels[y_pred_row > 0.21]) for y_pred_row in y_pred]

subm = pd.DataFrame()
subm['image_name'] = val_features.image_name.values
subm['tags'] = preds
subm.to_csv(submission_file, index=False)