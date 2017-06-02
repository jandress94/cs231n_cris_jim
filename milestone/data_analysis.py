import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import operator

train_labels = pd.read_csv('D:/cs231n_data/train_v2.csv')
# train_labels = pd.read_csv('/mnt/d/cs231n_data/train_v2.csv')

train_label_counter = Counter()

for row in train_labels.tags.values:

    train_label_counter.update(row.split())

print train_label_counter
sorted_labels = sorted(train_label_counter.items(), key=operator.itemgetter(1))

plt.bar(range(len(sorted_labels)), [x[1] for x in sorted_labels], align='center')
plt.xticks(range(len(sorted_labels)), [x[0] for x in sorted_labels], rotation=-90)
plt.ylabel('Number of Images')
plt.xlabel('Label')
plt.show()