import os
import numpy as np
from sklearn.cluster import KMeans

label_folder = "Dataset/labels/train"                       
n_anchors = 9

wh_list = []

for filename in os.listdir(label_folder):
    if filename.endswith(".txt"):
        with open(os.path.join(label_folder, filename), 'r') as f:
            for line in f:
                parts = line.strip().split()
                _, _, _, w, h = map(float, parts)
                wh_list.append([w,h])
                
wh_array = np.array(wh_list)
kmeans = KMeans(n_clusters=n_anchors, n_init="auto", random_state=42)
kmeans.fit(wh_array)
anchors = kmeans.cluster_centers_

anchors = sorted(anchors, key=lambda x : x[0]*x[1], reverse=True)

with open('initiative_anchors.txt', 'w') as f:
    for w,h in anchors:
        f.write(f"{w:.4f} {h:.4f}\n")