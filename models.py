#-*- coding:utf-8 -*-
#!/usr/bin/env python2.7
# ------------------------------------------------------------------------------
# Filename:    models.py
# Description: Model utilities
# ------------------------------------------------------------------------------

from __future__ import division
import csv
import numpy as np
import re
from sklearn.metrics import confusion_matrix

#Function that create a matrix X for machine learning issues
def create_mat(election,keys,name):
    flag = 1
    mat_file = 'data/output/features/'+election+'/'+name+'.csv'
    cr  = csv.reader(open(mat_file,'r'))
    cnt = 0
    L   = len(keys)
    for line in cr:
        if flag:
            X = np.zeros((L,len(line)-1))
            flag = 0
        key = line[0]
        if key in keys:
            X[keys.index(key),:] = line[1:]
            cnt = cnt + 1
    if cnt != L:
        print('Problem in matrix creation : '+str(cnt)+' lines in mat (required: '+str(L)+')')
        exit()
    return X


def homemade_getidx(info_set,reg):
    out_list = [info_set.index(re.match(reg,x).group())
    for x in info_set if re.match(reg,x) is not None]
    if not out_list:
        print('No element in info_set matches the regular expression %s' % reg)
        exit()
    else:
        return out_list

def create_ova(y,class_id):
    new_y = np.asarray([float('nan')]*len(y))
    class_idx = (y == i)
    new_y[class_idx] = 0
    new_y[~class_idx] = 1
    return new_y

def compute_best(y):
    y_unique = np.unique(y)
    best_freq   = 0
    best_class = -1
    for item in y_unique:
        freq = np.count_nonzero(y == item)
        if freq > best_freq:
            best_freq = freq
            best_class = item
    return best_class



# Return numpy array of errors
def compute_errors(y_valid, y_predict,error_names,nclass):
    cm        = confusion_matrix(y_valid, y_predict, range(nclass))
    errors    = np.empty((1+2*nclass,)) # Accuracy + (prec+rec)*nclass
    errors[:] = np.NAN

    for error_name in error_names:
        if error_name == 'accuracy':
            errors[0] = (sum(cm.diagonal())/cm.sum())
        elif error_name == 'precision':
            cm_norm = cm.astype('float') / cm.sum(axis=0)[np.newaxis,:] # PRECISION = VERTICAL
            #cm_norm[np.isnan(cm_norm)] =
            errors[1:1+nclass] = cm_norm.diagonal()
        elif error_name == 'recall':
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # RECALL = HORIZONTAL
            #cm_norm[np.isnan(cm_norm)] =
            errors[1+nclass:] = cm_norm.diagonal()

    return errors

def extract_info(info,info_set,X,keys,election):

    # Define some variables
    content_reg      = 'text_.*'
    follow_reg       = 'follow_nuts.*'+'|'+'follow_ca.*'
    all_reg          = 'text.*'+'|'+'follow.*'
    questions_reg    = 'questions'
    SP_reg           = 'SP'
    SP_questions_reg = 'SP'+'|'+'questions'

    # Model 0 : random
    flag = 0
    if info == 'random' or info == 'best' or info == 'real_random':
        #Xout   = np.NAN*np.empty((len(keys),1))
        Xout  = np.random.rand(len(keys),1)
        flag = 1
    else:
        # Model 1 : content aggregation
        if info == 'content':
            idx = homemade_getidx(info_set,content_reg)
        # Model 2 : follow aggregation
        elif info == 'follow':
            idx = homemade_getidx(info_set,follow_reg)
        # Model 3 : content aggregation + follow aggregation
        elif info == 'follow-content':
            perso_reg = follow_reg+'|'+content_reg
            idx   = homemade_getidx(info_set,perso_reg)
        # Model 4 : all = Twitter info  beginning by follow or text
        elif info == 'all':
            idx   = homemade_getidx(info_set,all_reg)
        # Model 5 : questions
        elif info == 'questions':
            idx   = homemade_getidx(info_set,questions_reg)
        # Model 6 : SP
        elif info == 'SP':
            idx   = homemade_getidx(info_set,SP_reg)
        # Model 7 : SP +
        elif info == 'SP-questions':
            idx   = homemade_getidx(info_set,SP_questions_reg)
        # Model 8 : content matrix
        elif info == 'content_mat':
            Xout = create_mat(election,list(keys),'text_wgt_2gram_users_mat')
            flag = 1
        # Model 9 : simple follow
        elif info == 'simple_follow':
            Xout = create_mat(election,list(keys),'simple_follow_users_mat')
            flag = 1
        # Model 10: follow-content_mat
        elif info == 'follow-content_mat':
            idx = homemade_getidx(info_set,follow_reg)
            Xout1 = X[:,idx]
            Xout2 = create_mat(election,list(keys),'text_wgt_2gram_users_mat')
            Xout  = np.concatenate((Xout1, Xout2), axis=1)
            flag  = 1
        # Model 11 : follow-SP-questions
        elif info == 'follow-SP-questions':
            perso_reg = follow_reg+'|'+SP_questions_reg
            idx   = homemade_getidx(info_set,perso_reg)
        # Model 12 : content-SP-questions
        elif info == 'content-SP-questions':
            perso_reg = content_reg+'|'+SP_questions_reg
            idx   = homemade_getidx(info_set,perso_reg)
        # Model 13 : follow-content-SP-questions
        elif info == 'follow-content-SP-questions':
            perso_reg = follow_reg+'|'+content_reg+'|'+SP_questions_reg
            idx   = homemade_getidx(info_set,perso_reg)
        else:
            print('Model '+info+' undefined')
            exit()

        # If output not else define, get specific indices
        if not flag:
            Xout = X[:,idx]
            if np.shape(Xout)[1] == 1:
                Xout = np.reshape(Xout,(np.shape(X)[0],1))
    return Xout

def learn(X_learn,y_learn,info,classificator,tools):
    if info == 'random':
        return None
    else:
        # Develop tools
        if classificator == 'ovo':
            if info == 'best':
                return [compute_best(y_learn)]
            else:
                tools[0].fit(X_learn, y_learn)
        elif classificator == 'ova':
            if info == 'best':
                bests = []
                for i in range(len(tools)):
                    bests.append(compute_best(y_learn_i))
            else:
                for i in range(len(tools)):
                    y_learn_i = create_ova(y_learn,i)
                    tools[i].fit(X_learn, y_learn_i)
        return tools
def predict_error(X_valid,y_valid,info,classificator,tools,nclass,error_names):
    if classificator == 'ovo':
        if info == 'random':
            y_predict = np.random.randint(nclass, size=len(y_valid))
        elif info == 'best':
            y_predict = np.asarray([tools[0]]*len(y_valid))
        else:
            y_predict  = tools[0].predict(X_valid)

        # Compute error and error average
        new_errors = compute_errors(y_valid, y_predict, error_names, nclass)

    elif classificator == 'ova':
        new_errors    = np.zeros((1,1+2*nclass))
        new_errors[0] = -1 # no accuracy info for ova
        for i in range(nclass):
            y_valid_i = create_ova(y_valid,i)
            if info == 'random':
                y_predict = np.random.randint(2, size=len(y_valid_i))
            elif info == 'best':
                y_predict = np.asarray([tools[i]]*len(y_valid_i))
            else:
                y_predict  = tools[i].predict(X_valid)

            # Compute error and error average
            new_errors_i  = compute_errors(y_valid_i, y_predict, error_names, 2)
            new_errors[i+1] = new_errors_i[1] # precision class 1
            errors_mat[i+1+nclass] = new_errors_i[3] # recall class 1
    return new_errors
