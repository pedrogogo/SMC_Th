import json
import numpy as np 
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import csv
#un-comment line below to print whole numpy arrays (note that computing cost will be much higher)
#np.set_printoptions(threshold=np.nan)

#this opens_loads the json files and assign them to their corresponding variables
dict_bright = json.load(open("bright_analysis.json", 'rb'))
dict_metal = json.load(open("metal_analysis.json",'rb'))
dict_hard = json.load(open("hard_analysis.json",'rb'))
dict_reverb = json.load(open("reverb_analysis.json",'rb'))
dict_rough = json.load(open("rough_analysis.json",'rb'))

#importing CSV files 
with open('bright.csv', 'rb') as f:
    reader = csv.reader(f)
    bright_list_csv = map(tuple, reader)
    #removing brackets from list
    bright_list = [l[0] for l in bright_list_csv]
    
with open('warm.csv', 'rb') as f:
    reader = csv.reader(f)
    warm_list_csv = list(reader)
    warm_list = [l[0] for l in warm_list_csv]
    
with open('rough.csv', 'rb') as f:
    reader = csv.reader(f)
    rough_list_csv = map(tuple, reader)
    rough_list = [l[0] for l in rough_list_csv]
    
with open('reverb.csv', 'rb') as f:
    reader = csv.reader(f)
    reverb_list_csv = map(tuple, reader)
    reverb_list = [l[0] for l in reverb_list_csv]
    
with open('clear.csv', 'rb') as f:
    reader = csv.reader(f)
    clear_list_csv = list(reader)
    clear_list = [l[0] for l in clear_list_csv]
    
with open('hollow.csv', 'rb') as f:
    reader = csv.reader(f)
    hollow_list_csv = list(reader)
    hollow_list = [l[0] for l in hollow_list_csv]
    
with open('deep.csv', 'rb') as f:
    reader = csv.reader(f)
    deep_list_csv = list(reader)
    deep_list = [l[0] for l in deep_list_csv]
    
with open('punchy.csv', 'rb') as f:
    reader = csv.reader(f)
    punchy_list_csv = list(reader)
    punchy_list = [l[0] for l in punchy_list_csv]
    
with open('metallic.csv', 'rb') as f:
    reader = csv.reader(f)
    metallic_list_csv = list(reader)
    metallic_list = [l[0] for l in metallic_list_csv]
    
with open('sharp.csv', 'rb') as f:
    reader = csv.reader(f)
    sharp_list_csv = list(reader)
    sharp_list = [l[0] for l in sharp_list_csv]
    
with open('hard.csv', 'rb') as f:
    reader = csv.reader(f)
    hard_list_csv = list(reader) 
    hard_list = [l[0] for l in hard_list_csv]

#printing to see content (change name to see contents. NOT MANDATORY. JUST PREVIEW)
hard_list
#check length, change to desired list
len(hard_list)

#Cleaning spurious 'nan' values from the list
for keys, values in clean_dict_bright.iteritems():
    print("BRIGHT-FreesoundID: %s VALUE: %s" % (keys, values))
    if dict_bright[keys] == 'nan':
        dict_bright.iteritems
    
for keys, values in dict_metal.iteritems():
    print("METAL-FreesoundID: %s VALUE: %s" % (keys, values))

for keys, values in dict_hard.iteritems():
    print("HARD VALUES-FreesoundID: %s VALUE: %s" % (keys, values))
    
    
for keys, values in dict_reverb.iteritems():
    print("REVERB VALUES-FreesoundID:%s VALUE: %s" % (keys, values))
    
for keys, values in dict_rough.iteritems():
    print("ROUGH VALUES-FreesoundID: %s VALUE: %s" % (keys, values))

#Confirming length
print "bright length:",len(dict_bright)
print "metal length:",len(dict_metal)
print "hard length:",len(dict_hard)
print "reverb length:",len(dict_reverb)
print "rough length:",len(dict_rough)

#cleaning nan again??? to be checked out
clean_dict_bright = filter(lambda k: not isnan(dict_bright[k]), dict_bright)
clean_dict_metal = filter(lambda k: not isnan(dict_metal[k]), dict_metal)
clean_dict_hard = filter(lambda k: not isnan(dict_hard[k]), dict_hard)
clean_dict_reverb = filter(lambda k: not isnan(dict_reverb[k]), dict_reverb)
clean_dict_rough = filter(lambda k: not isnan(dict_rough[k]), dict_rough)

#checking length of dictionaries
print "bright length:",len(clean_dict_bright)
print "metal length:",len(clean_dict_metal)
print "hard length:",len(clean_dict_hard)
print "reverb length:",len(clean_dict_reverb)
print "rough length:",len(clean_dict_rough)

#applying intersection to all the lists to ensure item are within all the lists
all_ids_intersection=list(set(clean_dict_bright) & set(clean_dict_metal) & set(clean_dict_hard) & set(clean_dict_rough))
all_ids_intersection
len(all_ids_intersection)

#creating matrix X
X = []

for fs_id in all_ids_intersection:
    feature_vector = [dict_bright[fs_id], dict_metal[fs_id], dict_hard[fs_id],dict_rough[fs_id]]
    X.append(feature_vector)

len(feature_vector) #checking length of the vector 
X = np.array(X)    
#previewing X
X
#checking out length
len(X)
#checking out shape of the matrix
X.shape

#creating y matrix
y = []
NB_SOUNDS = len(X)  #here will get same result if using "all_ids_intersection" instead of "X"
NB_LABELS = len(feature_vector)

y = np.zeros((NB_SOUNDS, NB_LABELS), dtype=int)

for idx, sound_id in enumerate(all_ids_intersection): # 
    if sound_id in bright_list: #if the sound is bright...
        y[idx][0] = 1 # add a 1 for each line (soundid) "idx" and the columns (label) 0....
    if sound_id in metallic_list: 
        y[idx][1] = 1 # add a 1 for each line (soundid) "idx" and the columns (label) 1....
    if sound_id in hard_list: 
        y[idx][2] = 1 # add a 1 for each line (soundid) "idx" and the columns (label) 2....
    if sound_id in rough_list: 
        y[idx][3] = 1 # add a 1 for each line (soundid) "idx" and the columns (label) 3....´
#checking y shape matrix
y.shape
#checking y content
y

#SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.80, random_state=42)
X_train.shape, y_train.shape
X_test.shape, y_test.shape

#this could have been done before.It is for double checking that non "nan" values are within the matrixes
print(np.isinf(X))
print(np.isnan(X))
print(np.isinf(y))
print(np.isnan(y))

#TRAIN
clf = OneVsRestClassifier(SVC(kernel='linear'))
clf.fit(X, y)

# #will need to check later on for optimization purposes, now seems the issue is somewhere else.
# 
# from sklearn.model_selection import GridSearchCV
# from sklearn import svm
# 
# parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
# svr = svm.SVC()
# 
# clf = GridSearchCV(svr,parameters)
# clf.fit(X,y)

#clf.predict_proba(X)  #not used yet

#checking out score and predict...need to be revised
clf.score(X, y, sample_weight=None)
clf.predict(X_test)

#THIS IS WRONG!!!!!!!!! NEED TO ITERATE OVER THE LISTS!!!!!!!!!
#The recall is the ratio tp / (tp + fn) 
#where tp is the number of true positives and fn the number of false negatives.
#The recall is intuitively the ability of the classifier to find all the positive samples.
#from sklearn.metrics import recall_score

###################need to iterate over the lists intead!!!!!!!!!!!#################
y_true = X_train.shape
y_pred = X_test.shape
 
x1 = recall_score(y_true, y_pred, average='micro') 
x2 = recall_score(y_true, y_pred, average='macro') 
x3 = recall_score(y_true, y_pred, average='weighted')
x1,x2,x3

#The recall is the ratio tp / (tp + fn) 
#where tp is the number of true positives and fn the number of false negatives.
#The recall is intuitively the ability of the classifier to find all the positive samples.
#from sklearn.metrics import recall_score
y_true = y_train.shape
y_pred = y_test.shape
m1 = recall_score(y_true, y_pred, average='micro')  
m2 = recall_score(y_true, y_pred, average='macro') 
m3 = recall_score(y_true, y_pred, average='weighted') 
m1, m2, m3

#The precision is the ratio tp / (tp + fp) 
#where tp is the number of true positives and fp the number of false positives. 
#The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.

#from sklearn.metrics import precision_score
y_true = y_train.shape
y_pred = y_test.shape
macro_prec=precision_score(y_true, y_pred, average='macro') 
weigh_prec= precision_score(y_true, y_pred, average='weighted')
micro_prec=precision_score(y_true, y_pred, average='micro')
#printing out values 
micro_prec, macro_prec, weigh_prec 

#from sklearn.metrics import accuracy_score
y_true = X_train.shape
y_pred = X_test.shape
accuracy_score(y_true, y_pred)

#TEST....classification report
#NEED TO CHECK PRECISION AND RECALL...RESULTS SEEM INCORRECT! (all 100%) NEED TO FIND OUT WHAT IS GOING ON!

#from sklearn.metrics import classification_report
#y_test = clf.predict(y_test) takes it from above
y_pred = clf.predict(X_test)
categories = ['bright', 'metal', 'hard', 'rough']
print(classification_report(y_test, y_pred, target_names=categories))

