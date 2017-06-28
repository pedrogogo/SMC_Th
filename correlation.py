import csv
import numpy as np
import json
import pandas as pd
import math
from scipy.stats import pearsonr


bright_sounds = []
hard_sounds = []
metal_sounds = []
reverb_sounds = []
rough_sounds = []

with open('csv_files/bright.csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        bright_sounds.append(row[0])
        
with open('csv_files/hard.csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        hard_sounds.append(row[0])
        
with open('csv_files/metallic.csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        metal_sounds.append(row[0])
        
with open('csv_files/reverb.csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        reverb_sounds.append(row[0])

with open('csv_files/rough.csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        rough_sounds.append(row[0])


bright_analysis = json.load(open('json_files/bright_analysis.json','rb'))
hard_analysis = json.load(open('json_files/hard_analysis.json','rb'))
metal_analysis = json.load(open('json_files/metal_analysis.json','rb'))
reverb_analysis = json.load(open('json_files/reverb_analysis.json','rb'))
rough_analysis = json.load(open('json_files/rough_analysis.json','rb'))

all_ids = list(set(bright_analysis.keys()) & set([h for h in hard_analysis.keys() if not math.isnan(hard_analysis[h])]) & set([m for m in metal_analysis.keys() if not math.isnan(metal_analysis[m])]) & set(reverb_analysis.keys()) & set(rough_analysis))

X = []
for id in all_ids:
    X.append([bright_analysis[id], hard_analysis[id], metal_analysis[id], reverb_analysis[id], rough_analysis[id]])
X = np.array(X)    
    
    
NB_SOUNDS = len(X)
NB_LABELS = len(X[0])
y = np.zeros((NB_SOUNDS, NB_LABELS), dtype=int)
for idx, sound_id in enumerate(all_ids): 
    if sound_id in bright_sounds: 
        y[idx][0] = 1 
    if sound_id in hard_sounds:
        y[idx][1] = 1
    if sound_id in metal_sounds: 
        y[idx][2] = 1 
    if sound_id in reverb_sounds: 
        y[idx][3] = 1 
    if sound_id in rough_sounds: 
        y[idx][4] = 1 

data = pd.DataFrame(np.concatenate((X,y), axis=1))

# get all correlations
corr = data.corr()

# one by one with the  p-value 
bright_2_bright_corr = pearsonr(X[:,0],y[:,0])




"""
PEARSON CORRELATION:
bright_descriptor hard_descriptor metal_descriptor reverb_descriptor rough_descriptor
bright_annotation hard_annotation metal_annotation reverb_annotation rough_annotation


          0         1         2         3         4         5         6         7         8         9  
0  1.000000  0.054169  0.614018  0.006531  0.358711  0.337453  0.001759  0.280689  0.001759 -0.182498  
1  0.054169  1.000000  0.028086 -0.006805  0.020261 -0.005260  0.003407  0.049588  0.003407 -0.017938  
2  0.614018  0.028086  1.000000  0.077347  0.338579  0.199428  0.060241  0.154477  0.060241 -0.034267  
3  0.006531 -0.006805  0.077347  1.000000 -0.075205  0.152128 -0.030763  0.037524 -0.030763  0.034147  
4  0.358711  0.020261  0.338579 -0.075205  1.000000  0.076449  0.010203  0.120501  0.010203  0.015494  
5  0.337453 -0.005260  0.199428  0.152128  0.076449  1.000000 -0.193827 -0.049328 -0.193827 -0.195646  
6  0.001759  0.003407  0.060241 -0.030763  0.010203 -0.193827  1.000000 -0.183563  1.000000 -0.218216  
7  0.280689  0.049588  0.154477  0.037524  0.120501 -0.049328 -0.183563  1.000000 -0.183563 -0.205427  
8  0.001759  0.003407  0.060241 -0.030763  0.010203 -0.193827  1.000000 -0.183563  1.000000 -0.218216  
9 -0.182498 -0.017938 -0.034267  0.034147  0.015494 -0.195646 -0.218216 -0.205427 -0.218216  1.000000

"""