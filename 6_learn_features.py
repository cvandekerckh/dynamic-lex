#-*- coding:utf-8 -*-
#!/usr/bin/env python2.7
# ------------------------------------------------------------------------------
# Filename:    3.3_machine_learning.py
# Description: Machine learning task cross-validation
#   classification - the precision/error recalls for each class
#   regression     - the regression error
# ------------------------------------------------------------------------------

from __future__ import division
import cPickle as pickle
import csv
import matplotlib.pyplot as plt
import numpy as np
from appendix import extract
from appendix import read_election
from models import extract_info, learn, predict_error
from sklearn import svm
from sklearn.ensemble import ExtraTreesClassifier,ExtraTreesRegressor
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from appendix import return_election

##########################################################################
############################ 1- PARAMETERS ###############################
##########################################################################

# Input Files
elections = return_election()
dic_path   = '../data/machine_learning/dictionaries/'

# Parameters
target   = 'users'

# General parameters
version        = 2                # v1 = simple accuracy, v2 = prec-recall
k_fold         = 3
predictor      = 'vote_intention' # ideology or vote_intention
n_iter         = 150               # Default : 150
kernel         = 'rbf'            # 'linear', 'rbf' #
gam            = 1/2              # 1/2 or 2
verbose_n      = 10               # display progression every n
np.random.seed = 1

# Output Files
output_path       = '../data/machine_learning/'

### Define features and methods
methods          = ['SVM']
classificators   = ['ovo']
if target == 'users':
    info_set = ['follow_nuts_party','text_wgt2_2gram','questions','SP',predictor]
    infos    = ['random','best','real_random','follow','content',
                'follow-content','questions','SP-questions',
                'follow-SP-questions','content-SP-questions',
                'follow-content-SP-questions']
elif target == 'candidates':
    info_set = ['follow_nuts','text_wf_2gram',predictor]
    infos    = ['real_random','follow','content','follow-content']


##########################################################################
############################# 2- FUNCTIONS ###############################
##########################################################################
def shift(l, n):
    return np.asarray(l[n:] + l[:n])

def homemade_zip(L1,L2):
    out = []
    for item1 in L1:
        for item2 in L2:
            out.append(str(item1)+'_'+str(item2))
    return out

def plot_confusion_matrix(cm, chosen_parties, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(chosen_parties))
    plt.xticks(tick_marks, chosen_parties, rotation=45)
    plt.yticks(tick_marks, chosen_parties)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

##########################################################################
############################# 3- EXECUTION ###############################
##########################################################################


### LOOP 1 : Over elections
for election in elections:

    print('Start with election '+election)

    # Read election
    chosen_parties = read_election(election,['major_parties'])[0].split(',')

    # Adapt follow_nuts et text_wgt to elections
    new_info_set      = list(info_set)

    # follow initialization choice
    if 'follow_nuts' in info_set:
        idx = info_set.index('follow_nuts')
        new_info_set[idx] = new_info_set[idx]+'_party'

    # Load dictionaries
    if target == 'users':
        dic_in       = dic_path+election+'/user_dic.p'
    elif target == 'candidates':
        dic_in       = dic_path+election+'/candidate_dic.p'
    dic  = pickle.load(open( dic_in, "rb" ))
    print(len(dic))

    # Header
    if predictor == 'vote_intention':
        chosen_class = chosen_parties
    elif predictor == 'ideology':
        chosen_class = ['left','center','right']
    nclass          = len(chosen_class)
    header_common   = ['election','method','classificator','predictor','info','n','learning_size',]
    header_n        = homemade_zip(['n'],chosen_parties)
    header_prec_rec = homemade_zip(['precision','recall'],chosen_class)
    header = header_common + header_n + ['accuracy'] + header_prec_rec

    ### LOOP 2 : Over methods
    for method in methods:
        print('Method '+method)
        ### LOOP 3 : Over classificators
        for classificator in classificators:
            print('Classificator '+str(classificator))
            tools = []
            if classificator == 'ovo':
                ntools = 1
            elif classificator == 'ova':
                ntools = nclass

            # Define classification methods
            for i in range(ntools):
                classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True))

            # Extract information from dictionary
            (keys,matrix) = extract(dic,new_info_set)
            n = len(keys)
            learning_size = int(round(n*(k_fold-1)/k_fold)) # learn set length
            ### LOOP 4 : Over information
            for info in infos:
                print('- Handle : %s' % info)
                y = matrix[:,new_info_set.index(predictor)]
                X = extract_info(info,new_info_set,matrix,keys,election)
                classes = np.unique(y)

                # Initialize output file
                header_v2 = ['n','learning_size']
                header_v2_info = [n,learning_size]
                y_valid_file = output_path+election+'_'+info+'_'+target+'_y-valid.csv'
                y_score_file = output_path+election+'_'+info+'_'+target+'_y-score.csv'
                with open(y_valid_file, 'w') as y_valid_out, open(y_score_file, 'w') as y_score_out :
                    cw_valid   = csv.writer(y_valid_out)
                    cw_valid.writerow(header_v2)
                    cw_valid.writerow(header_v2_info)
                    cw_valid.writerow(chosen_class)
                    cw_score   = csv.writer(y_score_out)
                    cw_score.writerow(header_v2)
                    cw_score.writerow(header_v2_info)
                    cw_score.writerow(chosen_class)


                ### START PREDICTION ###

                # 1) CV initialization
                cnt           = 0    # Iteration counter
                users_train   = keys # Training set permutation

                # 2) Define common data
                data_common   = [election,method,classificator,predictor,info,n,learning_size]
                data_n        = [y.tolist().count(i) for i in range(nclass)]

                ### Iteration loop
                for cnt in range(n_iter):
                    if cnt % verbose_n == 0:
                        print('Progression : %s/%s' % (str(cnt+1),n_iter))
                    # Permute users
                    users_train = np.random.permutation(users_train)

                    # Perform k-fold crossvalidation
                    for k in range(k_fold):

                        ###################################
                        #########  Learning stage #########
                        ###################################
                        # a) create sets
                        users_learn = users_train[1:learning_size]
                        idx         = [keys.tolist().index(x)
                                       for x in keys if x in users_learn]
                        y_learn     = y[np.asarray(idx)]
                        X_learn     = X[np.asarray(idx),:]

                        # b) Check that all class are represented in learning
                        if len(np.unique(y_learn)) < nclass:
                            continue

                        # c) Learn
                        y_learn   = label_binarize(y_learn,classes)

                        if nclass == 2:
                            # Homemade labelization
                            y_learn = np.concatenate((y_learn,~(y_learn)+2), axis=1)
                        tool_v2   = classifier.fit(X_learn, y_learn)

                        ###################################
                        #######  Prediction stage #########
                        ###################################
                        # a) create sets
                        users_cv    = users_train[learning_size:]
                        idx         = [keys.tolist().index(x)
                                       for x in keys if x in users_cv]
                        y_valid     = y[np.asarray(idx)]
                        X_valid     = X[np.asarray(idx),:]
                        # b) Check that all class are represented in validation

                        if len(np.unique(y_valid)) < nclass:
                            continue
                        # c) Predict and compute errors
                        if info == 'random':
                            y_score = 2*np.random.rand(len(y_valid),nclass)-1
                        elif info == 'best':
                            y_score = np.ones((len(y_valid),nclass))
                        else:
                            y_score = tool_v2.decision_function(X_valid)

                        with open(y_valid_file, 'a') as y_valid_out, open(y_score_file, 'a') as y_score_out :
                            cw_valid   = csv.writer(y_valid_out)
                            cw_score   = csv.writer(y_score_out)
                            y_valid_binarize = label_binarize(y_valid,classes)
                            if nclass == 2:
                                # Homemade labelization
                                y_valid_binarize = np.concatenate((y_valid_binarize,~(y_valid_binarize)+2), axis=1)
                            for i in range(np.shape(y_valid_binarize)[0]):
                                cw_valid.writerow(y_valid_binarize[i,:])
                                cw_score.writerow(y_score[i,:])

                        ########## Finally, shift the users by L
                        users_train = shift(users_train.tolist(),learning_size)

print('Success in machine learning')
