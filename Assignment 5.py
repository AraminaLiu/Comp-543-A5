# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 04:10:44 2021

@author: arami
"""
###TASK 1

import re
import numpy as np

# load up all of the 19997 documents in the corpus
corpus1 = sc.textFile ("s3://chrisjermainebucket/comp330_A5/SmallTrainingDataOneLinePerDoc.txt")
corpus2 = sc.textFile ("s3://chrisjermainebucket/comp330_A5/TrainingDataOneLinePerDoc.txt")
corpus3 = sc.textFile ("s3://chrisjermainebucket/comp330_A5/TestingDataOneLinePerDoc.txt")

def get_keyandlistofwords(corpus):
    validLines = corpus.filter(lambda x : 'id' in x)
    keyAndText = validLines.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:]))
    regex = re.compile('[^a-zA-Z]')
    keyAndListOfWords = keyAndText.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))
    return keyAndListOfWords

keyAndListOfWords2 = get_keyandlistofwords(corpus2)
allWords = keyAndListOfWords2.flatMap(lambda x: ((j, 1) for j in x[1]))

allCounts = allWords.reduceByKey (lambda a, b: a + b)

topWords = allCounts.top (20000, lambda x : x[1])

twentyK = sc.parallelize(range(20000))

dictionary = twentyK.map (lambda x: (topWords[x][0], x))

dictionary.lookup("applicant")
dictionary.lookup("and")
dictionary.lookup("attack")
dictionary.lookup("protein")
dictionary.lookup("car")

############Task 2


def convert_to_array(l):
    results = np.zeros(20000)
    for i in l:
        results[i] += 1
    return results


##if "Au" is in the key, y = 1, else y = 0
def convert_to_y(key):
    if 'AU' in key:
        return 1
    else:
        return 0
    
##compute tfidf rdd
def get_tfidf(corpus, numDocs):
    keyAndListOfWords = get_keyandlistofwords(corpus)
    word_id = keyAndListOfWords.flatMap(lambda x: ((j, x[0]) for j in x[1]))
    word_id_pos = word_id.join(dictionary)  #(word, (id, pos))
    id_pos = word_id_pos.values().groupByKey()
    id_freq = id_pos.map(lambda x:(x[0], convert_to_array(x[1]) ))
    tf = id_freq.map(lambda x: (x[0], x[1]/x[1].sum()))
    #numDocs = corpus.count()
    c1 = id_freq.map(lambda x: (x[0], np.clip(x[1], 0, 1)))
    c2 = c1.map(lambda x: (1, x[1]) )
    numDocswithWord = c2.reduceByKey(lambda a, b: a+b)
    idf = -np.ma.log(numDocswithWord.lookup(1)[0]/numDocs)
    idf = idf.filled(0)
    tf_idf = tf.map(lambda x: (x[0], x[1]*idf)).cache()
    return tf_idf

##Change key to y values
def get_tfidf_mod(corpus, numDocs):
    tf_idf = get_tfidf(corpus, numDocs)
    tf_idf_mod = tf_idf.map(lambda x: (convert_to_y(str(x[0])), x[1]))
    return tf_idf_mod
##compute mean
def get_mean(tf_idf_mod, numDocs):
    tf_idf_vals = tf_idf_mod.map(lambda x: (1, x[1]))
    tf_idf_sum = tf_idf_vals.reduceByKey(lambda a, b: a+b)
    tf_idf_mean = tf_idf_sum.lookup(1)[0]/numDocs
    return tf_idf_mean
##compute standard deviation    
def get_sd(tf_idf_mod, tf_idf_mean, numDocs):
    tf_idf_var = tf_idf_mod.map(lambda x: (1, (x[1] - tf_idf_mean)**2))
    tf_idf_var_sum = tf_idf_var.reduceByKey(lambda a, b: a+b)
    tf_idf_sd = np.sqrt(tf_idf_var_sum.lookup(1)[0]/numDocs)
    return tf_idf_sd
##compute normalized tfidf rdd    
def get_tfidf_norm(tf_idf_mod, tf_idf_mean, tf_idf_sd):
    tf_idf_norm = tf_idf_mod.map(lambda x:(x[0], (x[1] - tf_idf_mean)/tf_idf_sd))
    return tf_idf_norm

##Compute negative loglikelihood              
def llh(x, r):
    res= -x[0]*np.dot(x[1], r) + np.log(1+np.exp(np.dot(x[1], r)))
    return res
##Compute gradient
def gradient(x, r, beta, numDocs):
    res = -x[0]*x[1] + np.exp(np.dot(x[1], r))*x[1]/(1+np.exp(np.dot(x[1], r))) + 2*beta*r
    return res/numDocs


##optimize the gradient by gradient descent   
def gd_optimize(x, r, beta, numDocs):
    rate = 1
    llh1 = 1
    llh2 = 2
    while (abs(llh2 - llh1) > 10e-4):
        r_last = r 
        g = x.map(lambda x:(1, gradient(x, r, beta, numDocs))).reduceByKey(lambda a, b: a+b).lookup(1)[0]
        r = r - rate * g
        #print (f (x, y, w, c))
        llh1 = x.map(lambda x: (1, llh(x, r_last))).reduceByKey(lambda a, b: a+b).lookup(1)[0] + beta*np.linalg.norm(r_last)**2
        llh2 = x.map(lambda x: (1, llh(x, r))).reduceByKey(lambda a, b: a+b).lookup(1)[0] + beta*np.linalg.norm(r_last)**2
        if llh2 > llh1:
            rate = rate * .5
        else:
            rate = rate * 1.1
    return r

##small training data
numDocs1 = corpus1.count()
tfidf1_mod = get_tfidf_mod(corpus1, numDocs1)
mean = get_mean(tfidf1_mod, numDocs1)
sd = get_sd(tfidf1_mod, mean, numDocs1)
tfidf1 = get_tfidf_norm(tfidf1_mod, mean, sd)

##full training data
numDocs2 = corpus2.count()
tfidf2_mod = get_tfidf_mod(corpus2, numDocs2)
tfidf2 = get_tfidf_norm(tfidf2_mod, mean, sd)

##testing data
numDocs3 = corpus3.count()
tfidf3_mod = get_tfidf_mod(corpus3, numDocs3)
tfidf3 = get_tfidf_norm(tfidf3_mod, mean, sd)

##initialize r with random values, and compute gradient optimization on small training data
r_init = np.random.randn(20000)/10
r_start = gd_optimize(tfidf1, r_init, 0.01, numDocs1)

##use the optimized r for small training data as the initial values to train full data
r_computed = gd_optimize(tfidf2, r_start, 0.01, numDocs2)

##reverse position and word in dictionary
word_pos = dictionary.map(lambda x: (x[1], x[0]))
##print the 50 words with largest coefficients
top50 = np.argsort(r_start)[0:50]
for i in top50:
    word_pos.lookup(i)
    
#####Task 3
def predict(x, r):
    #compute number we predict to be true
    x_predict = x.map(lambda x: (1, np.dot(x[1], r))).cache()
    pred_vec = x_predict.lookup(1)
    predict_true = sum(i>0 for i in pred_vec)
    #compute number that are actually true
    real_T_rdd = x.map(lambda x: (1, x[0]))
    real_T_sum = real_T_rdd.reduceByKey(lambda a, b: a+b)
    real_true = real_T_sum.lookup(1)[0]
    #compute number we predict to be true that are actually true
    pred_real_rdd = x.map(lambda x: (1, x[0]*np.dot(x[1], r) )).cache()    #times the real classification, only true prediction of true cases will be positive
    pred_real = pred_real_rdd.lookup(1)
    T_pred_T = sum(i>0 for i in pred_real) 
    print('Number we predict to be Australian court cases: %f', predict_true)
    print('Number that are actually Australian court cases: %f', real_true)
    print('Number we say are court cases that are actually court cases: %f', T_pred_T)
    precision = T_pred_T/predict_true
    recall = T_pred_T/real_true
    print('Precision rate: %f', precision)
    print('Recall rate: %f', recall)
    F1 = 2*precision*recall/(precision + recall)
    print('F1 score is: %f',F1)

##get the F1 score
predict(tfidf3, r_computed)

def find_false_positive(x1, r):
    res = []
    y_rdd = x1.map(lambda x:(1, x[0]))
    y = y_rdd.lookup(1)
    x_predict = x1.map(lambda x: (1, np.dot(x[1], r))).cache()
    predict_val = x_predict.lookup(1)
    for i in range(len(y)):
        if (y[i] == 0 and predict_val[i] > 0):
            res.append(i)
    return res

##get three false positive articles
res = find_false_positive(tfidf3, r_computed)
validLines = corpus3.filter(lambda x : 'id' in x)
keyAndText3 = validLines.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:]))
for i in res[-3:]:
    keyAndText3.take(i)[i-1]