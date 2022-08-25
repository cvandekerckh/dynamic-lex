#-*- coding:utf-8 -*-
#!/usr/bin/env python2.7
# ------------------------------------------------------------------------------
# Filename:    appendix.py
# Description: Utility functions
# ------------------------------------------------------------------------------
import os
import csv
import matplotlib
import matplotlib.patches as patches
import matplotlib.path as path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random as rd
import re
from scipy.stats import spearmanr
from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA,FactorAnalysis
from sklearn.ensemble import ExtraTreesClassifier,ExtraTreesRegressor

#######################################################
################ 0 - Define outliers ##################
#######################################################

# Canada + Quebec outliers (due to non-correspondance with Twitter account)
outliers = [66371891,24738102] + [292220830] + [82242321,714452684,485296508,75019837,293281723]

#######################################################
#################### 1 -COLORS ########################
#######################################################

# Function that transform an RGB int into RGB triplet
def int2RGB(RGBint):
    RGBint = int(RGBint)
    Blue =  RGBint & 255
    Green = (RGBint >> 8) & 255
    Red =   (RGBint >> 16) & 255
    return(Red/255.0,Green/255.0,Blue/255.0)

def hex2int(hex_str):
    return int(hex_str.strip("#"),16)

def int2hex(RGBint):
    return '#'+str(format(int(RGBint), '06X'))

def colors_for_plot(color_list):
    color_list = [int2RGB(x) for x in color_list]
    color_list= np.reshape(np.asarray(color_list),(len(color_list),3))
    return color_list


#######################################################
#################### 2 - FILES ########################
#######################################################

# Extract files with 2 columns (ids, values):
# first line has to be a timestamp, not used here
# second line is the header, not used here
# extract the following lines in a tuple (id,values)

def create_data_struct(path,filename,elections):
    depth = 0
    with open(path + filename,'rU') as fn:
        last_elems = {}
        for line in fn:
            regline = re.match('(-*) (\w+)',line.rstrip('\n'))
            if regline is None:
                print('Problem in file %s with line %s' % (filename,line))
                exit()
            # Compute new tree ref
            new_depth = len(regline.group(1))
            last_elems[new_depth] = regline.group(2)
            if new_depth < depth:
                for k in range(new_depth+1,depth+1):
                    del last_elems[k]
            depth = new_depth
            # Convert to directory
            new_dir = '/'.join([last_elems[i+1] for i in range(len(last_elems))])
            #print(last_elems)
            if not os.path.exists(path + new_dir):
                os.makedirs(path + new_dir)
                if line.rstrip('\n').endswith('*'):
                    [os.makedirs('%s%s/%s' % (path,new_dir,election)) for election in elections]

    new_paths = [
        "../data/ideologies/",
        "../data/machine_learning/dictionaries/",
        "../data/machine_learning/ideologies/",
    ]

    for j in new_paths:
        tmp = os.listdir(j)
        for i in tmp:
            tmp_path = os.path.join(j, i, "details")
            if not os.path.exists(tmp_path):
                print(tmp_path)
                os.makedirs(tmp_path)

def file_to_tuple(filename):
    cr  = csv.reader(open(filename,'rU'))
    header = cr.next()
    id_idx = header.index('key')
    value_idx = header.index('value')
    ids    = []
    values = []
    for line in cr:
        ids.append(line[id_idx])
        values.append(line[value_idx])

    return (ids,values)

def file_to_dic(filename,key_str,value_str):
    cr  = csv.reader(open(filename,'rU'))
    header = cr.next()
    key_idx    = header.index(key_str)
    value_idx  = header.index(value_str)
    out_dic = {}
    for line in cr:
        out_dic[line[key_idx]] = line[value_idx]
    return out_dic

def create_codebook(vpl_codebook):
    cr               = csv.reader(open(vpl_codebook,'rU'),delimiter = ',')
    header           = cr.next()
    field_local_idx  = header.index('field_local')
    field_common_idx = header.index('field_common')
    data_type_idx    = header.index('data_type')
    in_dico_idx      = header.index('in_dico')

    codebook = {}
    for line in cr:
        codebook[line[field_local_idx]] = {
        'field_common':line[field_common_idx],
        'data_type'   :line[data_type_idx],
        'in_dico'     :line[in_dico_idx],
        }
    return codebook

def write_interesting(filename,fields,values):
    cw = csv.writer(open(filename,'w'))
    cw.writerow(fields) # Header


    values = zip(*values)
    for item in values:
        cw.writerow(item)

# Choose an election :
# input  : None
# output : chosen election by user

def return_election():
    election_file = '../data/election.csv'
    cr            = csv.reader(open(election_file,'rU'),delimiter = ',')
    header        = next(cr)
    election_idx  = header.index('election')
    elections     = []
    i             = 1
    for line in cr:
        election = line[election_idx]
        elections.append(line[election_idx])
        i = i + 1
    return elections

def choose_election():
    print('Choose election (enter number)?')
    election_file = '../data/election.csv'
    cr            = csv.reader(open(election_file,'rU'),delimiter = ',')
    header        = next(cr)
    election_idx  = header.index('election')
    elections     = []
    i             = 1
    for line in cr:
        election = line[election_idx]
        print(str(i)+' - '+election)
        elections.append(line[election_idx])
        i = i + 1
    elx = int(raw_input("> "))
    election = elections[elx-1]
    return election

def choose_target():
    print('Choose status : (1) politicians, (2) citizens')
    choice = int(raw_input("> "))
    if choice == 1:
        return 'politicians'
    elif choice == 2:
        return 'users'
    else:
        print('Wrong command')
        print('Entered command :'+str(choice))
        print('Expected command : 1 or 2')
        exit()



#######################################################
########### 3- Handling dictionaries ##################
#######################################################

# Function to add a row of values in dictionary
# Warning: newkeys and newvalues are returned popped!
def add_in_dic(dic,newkeys,newvalues,name):
    # a) Loop over dictionary and add matching values
    flag = 1 # To get the key-features
    key_features = []
    for key in dic:
        if flag:
            if dic[key] is not None:
                key_features = dic[key].keys()
            else:
                key_features = []
            flag = 0
        if key in newkeys:
            idx = newkeys.index(key)
            newvalue = newvalues.pop(idx)
            newkey   = newkeys.pop(idx)
            dic[newkey][name] = newvalue
        else:
            dic[key][name] = float('nan')
    # b) Append remaining values that didn't match
    for newkey in newkeys:
        dic[newkey] = dict.fromkeys(key_features,float('nan'))
        dic[newkey][name] = newvalues[newkeys.index(newkey)]

# Function that extract the common values in features
# with_nans = false ==> keep common non-nan features
# with_nans = true  ==> non-nan for mainfeat_list, incumb and isnan key of others
def extract(dic,features_name, with_nans = False, mainfeat_list = None):
    n = len(dic)
    # Extraction of only one element
    if not isinstance(features_name, list):
        matrix = np.zeros((n,1))
        matrix[:,0] = [dic[item][features_name] for item in dic]
        print(np.shape(matrix))

    # Extraction of multiple features
    else:
        m = len(features_name)
        matrix = np.zeros((n,m))
        i = 0
        for feature in features_name:
            matrix[:,i] = [dic[item][feature] for item in dic]
            i = i + 1

    # Handle nan entries
    if not with_nans:
        nonnan_values = ~(np.isnan(matrix).any(1))
    else:
        main_idx = [features_name.index(x) for x in mainfeat_list]
        nonnan_values = ~(np.isnan(matrix[:,main_idx]).any(1))
    keys   = np.asarray(dic.keys())[nonnan_values]
    matrix = matrix[nonnan_values]

    # Return keys and matrix
    # Eventually, incumb average values and add is_nan columns
    if not with_nans:
        out_tuple = (keys,matrix)
    else:
        are_nans      = np.zeros((np.shape(matrix)[0],np.shape(matrix)[1]-len(mainfeat_list)))
        are_nans_features = []
        incumb_cnt = 0
        for feature in features_name:
            if feature in mainfeat_list:
                continue
            else:
                i = features_name.index(feature)
                is_nan = np.isnan(matrix[:,i])
                matrix[is_nan,i] = np.nanmean(matrix[:,i])
                are_nans[:,incumb_cnt] = is_nan.astype(int)
                are_nans_features.append(feature+'_isnan')
                incumb_cnt = incumb_cnt + 1

        # Append new matrix to current matrix
        matrix = np.concatenate((matrix, are_nans), axis=1)
        final_features  = features_name + are_nans_features
        out_tuple = (keys,matrix,final_features)

    return out_tuple

#######################################################
#################### 4 - Filter entries ###############
#######################################################

# 2) Functions
# 2a) Convert list of string to floats
# isdigit function for floats
def isdigit2(txt):
    try:
        float(txt)
        return True
    except ValueError:
        return False

# Convert list to digit, and put nan/unchange others
def todigit(prev_list, keep_strings = False):
    new_list = prev_list
    cnt = 0
    for x in new_list:
        # Converts digitable strings to digits
        if isdigit2(x):
            new_list[cnt] = float(x)
        else:
            if x == 'NA' or not keep_strings:
                new_list[cnt] = float('nan')
            else:
                new_list[cnt] = x
        cnt = cnt+1
    return new_list

# 2b) Convert incomeRaw to float
def income_to_float(income_list):
    float_list = income_list
    cnt = 0
    for item in income_list:
        if item == 'NA':
            float_list[cnt] = float('nan')
        else:
            income_re  = re.compile('|'.join([r'\$(\d+)-',r'\$(\d+)']))
            income_re_mil = re.compile(r'\$(\d+)\s') # For 1 million
            match = income_re.match(item)
            if match is None:
                match_million = income_re_mil.match(item)
                if match_million is None:
                    print('Problem for matching '+item)
                    exit()
                else:
                    float_list[cnt] = 1000.0
            else:
                if match.group(1) is None:
                    float_list[cnt] = float(match.group(2))
                else:
                    float_list[cnt] = float(match.group(1))
        cnt = cnt + 1
    return float_list

def spec_to_float(field,spec_list):
    float_list = spec_list
    cnt = 0
    for item in spec_list:
        float_list[cnt] = specific_dic[field][item]
        cnt = cnt + 1
    return float_list

# 2c) Filter labels (before label creation) and feature (for feature selection)
def filter_entries(VPL_dic,codebook,chosen_parties):
    for field in codebook:
        in_dico      = codebook[field]['in_dico']
        # Filter only existing entries
        if in_dico == '1':
            field_common = codebook[field]['field_common']

            # A. Ordinal : convert to digits
            if codebook[field]['data_type'] == 'ordinal':
                VPL_dic[field_common] = todigit(VPL_dic[field_common])
            # B. Categorical: expand to higher dimension
            elif codebook[field]['data_type'] == 'categorical':
                categories = VPL_dic[field_common]
                categories_array = pd.get_dummies(pd.Series(categories))
                del VPL_dic[field_common] # suppress the field
                for cat_name in categories_array.keys(): # replace by categories
                    VPL_dic[field_common+'__'+cat_name] = categories_array[cat_name].values.tolist()

            # C. Specific : use specific_dic
            elif codebook[field]['data_type'] == 'specific':
                if field_common == 'incomeRaw':
                    VPL_dic[field_common] = income_to_float(VPL_dic[field_common])
                elif (field_common == 'vote_intention') or (field == 'pvote_intetion'):
                    VPL_dic[field_common] = [chosen_parties.index(x)
                    if x in chosen_parties else float('nan')
                    for x in VPL_dic[field_common]]
                else:
                    VPL_dic[field] = spec_to_float(field,VPL_dic[field])
            else:
                print('Problem with "typeof" of field '+field)

    return VPL_dic

#######################################################
################## 5 - Dependent Fields ###############
#######################################################

def CronbachAlpha(itemscores):
    itemscores = np.asarray(itemscores)
    itemvars = itemscores.var(axis=1, ddof=1)
    tscores = itemscores.sum(axis=0)
    nitems = len(itemscores)

    return nitems / (nitems-1.) * (1 - itemvars.sum() / tscores.var(ddof=1))

def create_dependent_fields(user_dic,predicted_field,nquestions):

    if predicted_field == 'questions':
        # Initialization
        fa = FactorAnalysis()

        # Create matrix for factorial analysis
        fields = []
        for i in range(nquestions):
            field = 'q'+str(i+1)
            fields.append(field)

        (keys,X) = extract(user_dic,fields)

        # Center data
        mu_questions  = np.tile(np.mean(X,axis = 0),(np.shape(X)[0], 1))
        std_questions = np.tile(np.std(X,axis = 0),(np.shape(X)[0],1))
        X_cen         = np.divide((X - mu_questions),std_questions)

        # Factor Analysis on left-right dimension
        fa = FactorAnalysis(n_components=1)
        fa.fit(X_cen)
        weights = fa.components_[0]
        y = fa.transform(X_cen)[:,0]

        # Eigenvalues
        print('Eigenvalues : ')
        eig = np.linalg.eig(np.corrcoef(np.transpose(X_cen)))[0].tolist()
        eig = [x.real for x in eig]
        eig = sorted(eig,reverse = True)
        print(eig)
        #plt.figure()
        #plt.title("Eigenvalues")
        #plt.bar(range(len(eig)), eig,
                  #color="r", align="center")
        #plt.show()

        # Cronback Alpha
        X_cen[:,np.sign(weights) == -1] = -X_cen[:,np.sign(weights) == -1]
        itemscores = np.transpose(X_cen)
        print("Cronbach alpha = ", CronbachAlpha(itemscores))

        print('FA weights:')
        print(weights)

        # Add info in dictionary the questions aggregation
        keys_new = list(keys.tolist())
        y_new    = list(y.tolist())
        add_in_dic(user_dic,keys_new,y_new,'questions')

        # Classify ideologies
        ncuts = 3.0
        cuts = [np.percentile(y, 100*k/ncuts) for k in range(int(ncuts)) if k > 0]
        idx_array = np.zeros((int(ncuts),len(y)),dtype=bool)
        for i in range(int(ncuts)):
            if i == 0:
                idx_array[i,:] = (y < cuts[i])
            elif i == ncuts-1:
                 idx_array[i,:] = (y >= cuts[i-1])
            else:
                 idx_array[i,:] = (y >= cuts[i-1]) & (y < cuts[i])
        for i in range(int(ncuts)):
            y[idx_array[i,:]] = i
        add_in_dic(user_dic,keys.tolist(),y.tolist(),'ideology')


    elif predicted_field == 'partyxA': # NOT YET EFFECTIVE
        # a) Initialize best parties
        best_parties  = [0]*L
        best_scores   = [-1]*L
        labels        = [0]*L

        # b) Find the VPLdigit that maximize the score
        # Randomize order of parties (in case of equality maximum)
        n_prefparties = len(VPLdigit_to_party.keys())-1 # Suppress NAN
        perm = np.random.permutation(n_prefparties)

        # Find best party
        for i in perm:
            field = 'party'+str(i+1)+'A'
            scores = todigit(user_dic[field])
            score_cnt = 0
            for value in scores:
                if value > best_scores[score_cnt]:
                    best_scores[score_cnt]  = value
                    best_parties[score_cnt] = i+1
                score_cnt = score_cnt + 1

        # c) Convert the best parties to labels
        party_cnt = 0
        for party in best_parties:
            party_name = VPLdigit_to_party[party]
            if party_name in party_dic.keys():
                labels[party_cnt] = party_dic[party_name]['ID']
            else:
                labels[party_cnt] = float('nan')
            party_cnt = party_cnt + 1

    elif predicted_field == 'random': # NOT YET EFFECTIVE
        L = len(user_dic['theta'])
        possible_labels = []
        for field in party_dic:
            possible_labels.append(party_dic[field]['ID'])
        labels = [rd.sample(possible_labels,1)[0] for predict in range(L)]
    else:
        print('Error in predicted field entry')
        exit()

#######################################################
##################### 6 - PLOTS #######################
#######################################################

# Choose two fields to plot (option : add a third field for color)
def add_plot(dic,fields,title = None, inverse_x = False, inverse_y = False,to_display = None, xlabl = None, ylabl = None, scale = True, common_fields = None,
filter_outliers = False, tmp_file = None):

    if common_fields == None:
        (keys,matrix) = extract(dic,fields)
    else:
        (keys,matrix) = extract(dic,common_fields)
    print('nsamples : '+str(len(keys)))

    # Filter outliers
    if filter_outliers:
        keys = np.asarray([float(key) if int(key) not in outliers
        else float('nan') for key in keys])
        nonnan_values = ~(np.isnan(keys))
        keys   = keys[nonnan_values]
        matrix = matrix[nonnan_values,:]

    # Define colors
    if common_fields is not None:
        my_colors = matrix[:,common_fields.index('colors')]
        colors_plot = colors_for_plot(my_colors)
    elif 'colors' in fields:
        my_colors = matrix[:,fields.index('colors')]
        colors_plot = colors_for_plot(my_colors)
    else:
        colors_plot = 'black'

    # Define sizes
    if 'followers_count' in fields:
        followers_counts = matrix[:,fields.index('followers_count')]
        my_sizes         = list(followers_counts)
        surf_min = 150
        surf_max = 800
        my_min = min(my_sizes)
        my_max = max(my_sizes)
        my_sizes = [int((my_size - my_min)*(surf_max-surf_min)/(my_max-my_min) + surf_min) for my_size in my_sizes] # mapping from one interval to another
        my_sizes         = np.asarray(my_sizes)
        followers_counts = np.asarray([int(x) for x in followers_counts])
    else:
        my_sizes = np.asarray([700]*len(keys))
        followers_counts = my_sizes


    # Define plot vectors
    if common_fields == None:
        x1 = pow(-1,inverse_x)*matrix[:,0]
        x2 = pow(-1,inverse_y)*matrix[:,1]
    else:
        x1 = pow(-1,inverse_y)*matrix[:,common_fields.index(fields[0])]
        x2 = pow(-1,inverse_y)*matrix[:,common_fields.index(fields[1])]
    rho = np.corrcoef(x1,x2)

    if scale:
        x1 = preprocessing.scale(x1)
        x2 = preprocessing.scale(x2)

    print('Correlation '+str(fields[0])+'-'+str(fields[1]))
    print('Pierson: '+str(rho[0,1]))
    print('Spearman: '+str(spearmanr(x1,x2)[0])+'\n')
    plt.grid('on')
    plt.scatter(x1,x2,alpha=0.9,s = my_sizes,facecolors=colors_plot)

    if to_display is not None:
        twitter_ids = [int(x) for x in to_display.keys() if x.isdigit()]
        labels_idx = [keys.tolist().index(x) for x in twitter_ids if x in keys]

        x1_lab               = x1[labels_idx]
        x2_lab               = x2[labels_idx]
        my_sizes_lab         = my_sizes[labels_idx]
        followers_counts_lab = followers_counts[labels_idx]
        my_colors_lab        = my_colors[labels_idx]
        new_keys             = keys[labels_idx]
        # print(new_keys[0:3])
        # print(x1_lab[0:3])
        # print(x2_lab[0:3])
        labels = [to_display[str(int(x))] for x in new_keys]
        print(labels)
        for label, x, y in zip(labels, x1_lab, x2_lab):
            plt.annotate(
                label,
                xy = (x, y), xytext = (0,25),
                #xytext = (-20, 20),
                textcoords = 'offset points', ha = 'center', va = 'center',fontsize=20, # 8?
                #bbox = dict(boxstyle = 'round,pad=0.5', fc = 'none', alpha = 0.5),
                arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0')
                )
    # Axis labels
    fs = 30
    if xlabl == None:
        plt.xlabel(fields[0]+' ideal points',fontsize = fs)
    else:
        plt.xlabel(xlabl,fontsize = fs)
    if ylabl == None :
        plt.ylabel(fields[1]+' ideal points', fontsize = fs)
    else:
        plt.ylabel(ylabl, fontsize = fs)

    if title is not None:
        plt.title(title)
    #plt.grid(True)
    eps = 0.4
    plt.xlim([np.min(x1)-eps, np.max(x1)+eps])
    plt.ylim([np.min(x2)-eps, np.max(x2)+2*eps])
    plt.xticks(np.linspace(-2,2,5), color='k', size=30)
    plt.yticks(np.linspace(-2,2,5), color='k', size=30)
    plt.tick_params(axis='both', which='major', pad=10)
    plt.tight_layout()

    if tmp_file is not None:
        if to_display is not None:
            write_names = (len(labels) == len(x1))
        else:
            write_names = False

        with open(tmp_file,'w') as f:
            cw = csv.writer(f)
            header = ['Twitter_ID','net','txt','colors']
            if write_names:
                header = header+['name']
            if 'followers_count' in fields:
                header = header + ['followers_count','sizes']
            cw.writerow(header)
            for i in range(len(keys)):
                if write_names:
                    to_write = [int(new_keys[i]),x1_lab[i],x2_lab[i],my_colors_lab[i],labels[i]]
                    if 'followers_count' in fields:
                        to_write = to_write + [followers_counts_lab[i]]
                        to_write = to_write + [my_sizes_lab[i]]
                else:
                    to_write = [int(keys[i]),x1[i],x2[i],my_colors[i]]
                cw.writerow(to_write)


def plot_hist(ax, array, colors, hatches, elections, infos, std_array = None):

    # Parameters
    extract_baselines = False
    #space_positions = [2,5]
    space_positions = []
    if std_array is None:
        std_array = np.NAN * np.empty(np.shape(new_array))

    # Extract baselines
    if extract_baselines:
        b1 = 'Random'
        b2 = 'Best'
    else:
        b1 = ''
        b2 = ''
    idx1 = (np.asarray(infos) == b1) #'Random', 'Best' or ''
    idx2 = (np.asarray(infos) == b2)
    idx3 = ~(idx1 | idx2)
    baseline_1 = array[idx1,:]
    baseline_1_std = std_array[idx1,:]
    baseline_2 = array[idx2,:]
    baseline_2_std = std_array[idx2,:]
    new_array  = array[idx3,:]
    new_std_array = std_array[idx3,:]
    new_colors    = np.asarray(colors)[idx3]
    new_hatches   = np.asarray(hatches)[idx3]
    new_infos     = np.asarray(infos)[idx3]
    (M,N) = np.shape(new_array)


    ## necessary variables
    ind = np.arange(N)                # the x locations for the groups
    width = 1.0/(M+2+len(space_positions))        # Considering L-white bar + two bar space

    ## the bars
    rects = []
    spaces = [sum(np.asarray([x]*len(space_positions))>= space_positions)
                   for x in range(M)]
    for i in range(M):
        rects.append(ax.bar(ind+i*width+ spaces[i]*width, new_array[i,:], width,
                        color=new_colors[i],
                        hatch=new_hatches[i],
                        yerr=new_std_array[i,:],
                        error_kw=dict(elinewidth=2,ecolor='black')))

    if extract_baselines:
        for i in range(N):
            plt.plot([ind[i],ind[i]+(len(new_infos)+1)*width],[baseline_1[:,i]]*2,color = 'red',linewidth = 3)
            plt.plot([ind[i],ind[i]+(len(new_infos)+1)*width],[baseline_2[:,i]]*2,'r--',linewidth = 3)

    # axes and labels
    ax.set_xlim(-width,len(ind)+width)
    ax.set_ylim(0,130)
    ax.set_ylabel('Accuracy',fontsize=14)
    ax.set_title('Vote intention : prediction accuracy')
    xTickMarks = elections#['Group'+str(i) for i in range(1,N+1)]
    ax.set_xticks(ind+(M*width/2.0))
    ax.set_yticks([0,20,40,60,80,100])
    ax.yaxis.grid()
    xtickNames = ax.set_xticklabels(xTickMarks)
    plt.setp(xtickNames, rotation=0, fontsize=14)

    ## add a legend
    ax.legend( rects, new_infos,
    ncol = 3
     )

def plot_prec_rec(ax,prec_rec_array,party_names,colors,elections):
    x1 = prec_rec_array[0,:]
    x2 = prec_rec_array[1,:]
    for_leg = []
    colors = np.asarray(colors)
    colors_unique = pd.unique(colors)
    for color in colors_unique:
        idx = (colors == color)
        for_leg.append(plt.scatter(x1[idx],x2[idx], c = color, s = 80))
    for label, x, y in zip(party_names, x1, x2):
        plt.annotate(
            label,
            xy = (x, y), xytext = (0, 13),
            textcoords = 'offset points', ha = 'center', va = 'center',fontsize=12
            )
    ax.set_ylim(0,100)
    ax.set_xlim(0,100)
    #ax.legend(for_leg,elections,scatterpoints=1,loc='lower right',ncol=2,fontsize=14)
    ax.grid(which='major', alpha=0.5)
    ax.set_title('Vote intention : precision and recall (Network + Text)')
    ax.set_ylabel('Recall',fontsize=14)
    ax.set_xlabel('Precision',fontsize=14)
    ax.set_ylim(0,110)
    #ax.grid(which='major', alpha=0.5)

def plot_matrix(matrix, fields, title=None, cmap=plt.cm.Greens):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    n = np.shape(matrix)[0]
    plt.imshow(matrix, interpolation='nearest', cmap=cmap)

    #### CREATE LABELS ###
    cnt = 0
    labels = []
    x1_lab = []
    x2_lab = []
    for label in np.nditer(matrix):
        labels.append(round(float(label),2))
        x1_lab.append(cnt/n)
        x2_lab.append(cnt % n)
        cnt = cnt + 1
    #####################

    for label, x, y in zip(labels, x1_lab, x2_lab):
        if x != y:
            ax.annotate(label,xy = (x, y), xytext = (x-0.2,y),fontsize=20)
    #plt.title(title)
    #plt.colorbar()
    tick_x = np.arange(n)
    tick_y = np.arange(n)
    plt.xticks(tick_x, fields, fontsize=17)
    plt.yticks(tick_y, fields, fontsize=17)
    ax.tick_params(axis='both', which='major', pad=8)
    #plt.tight_layout()
    #plt.ylabel('True label')
    #plt.xlabel('Predicted label')


def select_features(X,y,n,keys,pred_type):

    # Parameter
    string_limit = 6
    nusers       = np.shape(X)[0]
    predictor    = keys[-1]
    features     = keys[0:-1]

    # Build a forest and compute the feature importances
    if pred_type == 'class':
        forest = ExtraTreesClassifier(n_estimators=250,
                                  random_state=0)
    elif pred_type == 'reg':
        forest = ExtraTreesRegressor(n_estimators=250,
                                  random_state=0)
    else:
        print('Problem with prediction type')
        exit()

    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %s (%f)" % (f + 1, features[indices[f]][0:string_limit], importances[indices[f]]))
        if (f+1) > n:
            break

    ticks = [features[indices[x]][0:string_limit] for x in range(n)]
    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances for "+predictor+" - "+str(nusers)+" users")
    plt.bar(range(X.shape[1])[0:n], importances[indices][0:n],
           color="r", yerr=std[indices][0:n], align="center")
    plt.xticks(range(X.shape[1])[0:n], ticks)
    plt.xlim([-1, n])
    plt.show()

def read_election(election,fields,election_file = '../data/election.csv'):
    # Initialization
    out_list       = []
    flag           = 0
    field_indices  = []

    # Declare reader
    cr       = csv.reader(open(election_file,'rU'),delimiter = ',')

    # Handle header
    header   = cr.next()
    election_idx = header.index('election')
    for field in fields:
        field_indices.append(header.index(field))

    # Extract fields
    for line in cr:
        if line[election_idx] == election:
            flag = 1
            for field_idx in field_indices:
                out_list.append(line[field_idx])
    # Return queries
    if flag == 1:
        return out_list
    else:
        raise('Error in read_election : election not found')

def plot_kiviat(properties,values,cnt,n,offset):
    # Data to be represented
    # ----------
    # properties = ['property 1', 'property 2', 'property 3']
    # values = np.random.uniform(5,9,len(properties))
    # ----------

    # Use a polar axes
    axes = plt.subplot(1,n,cnt, polar=True)

    # Set ticks to the number of properties (in radians)
    t = np.arange(np.pi/2+offset,2*np.pi+np.pi/2+offset,2*np.pi/len(properties))

    plt.xticks(t, [])

    # Set yticks from 0 to 10
    plt.yticks(np.linspace(0,100,6),[])
    #plt.grid(linestyle=':')

    #plt.grid('off')


    # Draw polygon representing values
    points = [(x,y) for x,y in zip(t,values)]
    points.append(points[0])
    points = np.array(points)
    codes = [path.Path.MOVETO,] + \
            [path.Path.LINETO,]*(len(values) -1) + \
            [ path.Path.CLOSEPOLY ]
    _path = path.Path(points, codes)
    _patch = patches.PathPatch(_path, fill=True, color='red', linewidth=0, alpha=.1)
    axes.add_patch(_patch)
    _patch = patches.PathPatch(_path, fill=False, color='red', linewidth = 2)
    axes.add_patch(_patch)

    # Draw circles at value points
    plt.scatter(points[:,0],points[:,1], linewidth=2,
                s=100, color='red', edgecolor='red', zorder=10)

    # Set axes limits
    plt.ylim(0,100)

    # Properties labels
    for i in range(len(properties)):
        angle_rad = i/float(len(properties))*(2*np.pi)+np.pi/2+offset
        #angle_deg = i/float(len(properties))*360
        ha = "right"
        if angle_rad < np.pi/2 or angle_rad > 3*np.pi/2: ha = "left"
        if angle_rad == np.pi/2 or angle_rad == 3*np.pi/2: ha = "center"
        plt.text(angle_rad, 110, properties[i], size=16,
                 horizontalalignment=ha, verticalalignment="center")

    # Value labels
    for i in range(len(properties)):
        eps = 10
        angle_rad = i/float(len(properties))*(2*np.pi)+np.pi/2+offset
        eps_theta = np.pi/5-angle_rad/10

        #angle_deg = i/float(len(properties))*360
        ha = "right"
        if angle_rad < np.pi/2 or angle_rad > 3*np.pi/2: ha = "left"
        if angle_rad == np.pi/2 or angle_rad == 3*np.pi/2: ha = "center"
        plt.text(angle_rad+eps_theta, values[i]+eps, str(round(values[i]/100,2)), size=14,
                 horizontalalignment=ha, verticalalignment="center")
