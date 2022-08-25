#-*- coding:utf-8 -*-
#!/usr/bin/env python2.7
# ------------------------------------------------------------------------------
# Filename:    5_aggregate_features.py
# Description: Creates dictionaries for users and candidates
#              Easy to handle missing data and mixing Twitter and VPL values
# Warning:     Assert that no feature field in VPL with '__' in name
# ------------------------------------------------------------------------------
from __future__ import division
import cPickle as pickle
import csv
import math
import numpy as np
import os
import re
from appendix import *
from matplotlib.colors import BoundaryNorm
from appendix import choose_election, return_election

elections = return_election()

##########################################################################
############################ 1- PARAMETERS ###############################
##########################################################################

# Options
tasks  = ['DESC','ML','FILTERS'] # ML or DESC
modes = [3,2,3] # 3 : users + pols, 2 : users, 1: politicians

for election in elections:
    print('Aggregation for %s' % election)
    for kk,task in enumerate(tasks):
        mode = modes[kk]

        if task == 'ML':
            general_path = '../data/machine_learning/'
        elif task == 'DESC':
            general_path = '../data/'
        elif task == 'FILTERS':
            general_path = '../data/filters/'

        path             = general_path + 'ideologies/'+election+'/'
        user_dic_file    = general_path + 'dictionaries/'+election+'/user_dic.p'

        # Input files
        vpl_file_user    = '../data/raw/vpl/'+election+'-extended.csv'
        vpl_file_pol     = '../data/raw/vpl/'+election+'-politicians-extended.csv'
        vpl_codebook     = '../data/raw/vpl/'+election+'-codebook.csv'


        candidate_dic_file        = general_path + 'dictionaries/'+election+'/candidate_dic.p'
        candidate_to_display_file = general_path + 'dictionaries/'+election+\
                                    '/candidate_to_display_dic.p'

        ## User Specific Parameter
        nquestions            = 30               # Number of questions for factorial analysis
        color_field           = 'vote_intention' #'undecided__Undecided' #
        every_n               = 10               # Print progression for user_dic creation

        ##########################################################################
        ############################# 2- EXECUTION ###############################
        ##########################################################################

        # Get feature files
        users_files       = []
        politicians_files = []
        for file in os.listdir(path):
            if file.endswith("_politicians.csv"):
                politicians_files.append(file)
            elif file.endswith("_users.csv"):
                users_files.append(file)

        # Get election features
        [abrev_parties,chosen_parties,colors_parties,names_to_display,surnames_to_display,major_parties] = read_election(election,['party_abr','party_to_display','party_colors','names_to_display','surnames_to_display','major_parties'])
        abrev_parties           = abrev_parties.split(',')
        chosen_parties          = chosen_parties.split(',')
        candidate_to_display    = names_to_display.split(',')
        surcandidate_to_display = surnames_to_display.split(',')
        major_parties           = major_parties.split(',')

        colors_parties   = colors_parties.split(',')
        colors_parties   = [colors_parties[i] for i in range(len(colors_parties))
                            if abrev_parties[i] in chosen_parties]

        # Choose to build a dic or use a previous one
        print('Create a new dictionary for task %s' % task)
        #mode = int(raw_input("> "))

        #################
        # CANDIDATE DIC #
        #################


        if mode == 1 or mode == 3:

            # 1) Initialize dictionaries
            candidate_dic = {}
            candidate_to_display_dic = {}

            # 2) Add features to dictionary
            for filename in politicians_files:
                feature      = re.sub('_politicians.csv', '', filename)
                (ids,values) = file_to_tuple(path+filename)
                add_in_dic(candidate_dic,ids,values,feature)
            if task == 'FILTERS': ## Add baseline ideologies for control
                feature = 'follow_nuts_party'
                (ids,values) = file_to_tuple('../data/ideologies/%s/follow_nuts_party_politicians.csv' % election)
                add_in_dic(candidate_dic,ids,values,feature)

            # 3) Associate colors, vote_intentions (candidate_dic)
            # and names (candidate_to_display_dic)
            cr_pol              = csv.reader(open(vpl_file_pol,'rU'))
            header_pol          = cr_pol.next()
            ID_idx              = header_pol.index('Twitter_ID')
            name_idx            = header_pol.index('Twitter_account')
            party_idx           = header_pol.index('Party')
            Name_idx = False
            twitter_candidates     = [] # candidates associated with parties
            twitter_candidates_all = []
            colors_candidates      = []
            vote_intentions        = []
            followers_counts       = []
            for line in cr_pol:
                party   = line[party_idx]
                twitter = line[ID_idx]
                twitter_candidates_all.append(twitter)
                name  = line[name_idx]

                # Add in candidate_dic
                if (party in chosen_parties) and (twitter != 'NA'):
                    color = colors_parties[chosen_parties.index(party)]
                    color = hex2int(color)
                    twitter_candidates.append(twitter)
                    colors_candidates.append(color)
                    vote_intentions.append(chosen_parties.index(party))

                else:
                    if name in candidate_to_display:
                        sur_idx = candidate_to_display.index(name)
                        candidate_to_display_dic[twitter] = surcandidate_to_display[sur_idx]

            add_in_dic(candidate_dic,list(twitter_candidates),colors_candidates,'colors')
            add_in_dic(candidate_dic,list(twitter_candidates),vote_intentions,'vote_intention')

            # 4) Save dictionaries
            pickle.dump(candidate_dic, open(candidate_dic_file, "wb" ) )
            if task == 'DESC':
                pickle.dump(candidate_to_display_dic, open(candidate_to_display_file, "wb" ) )
            print('Candidate dictionary written with success')
            #print candidate_dic[candidate_dic.keys()[1]]

        #################
        #### USER DIC ###
        #################

        if mode == 2 or mode == 3:

            # 1) Add Twitter in a User dictionary
            # Initialize dictionary
            user_dic    = {}

            for filename in users_files:
                feature   = re.sub('_users.csv', '', filename)
                (ids,values) = file_to_tuple(path+filename)
                add_in_dic(user_dic,ids,values,feature)
            if task == 'FILTERS': ## Add baseline ideologies for control
                feature = 'follow_nuts_party'
                (ids,values) = file_to_tuple('../data/ideologies/%s/follow_nuts_party_users.csv' % election)
                add_in_dic(user_dic,ids,values,feature)

            # 2) Create a VPL dictionary
            # 2.1 - Open VPL codebook and create codebook
            codebook = create_codebook(vpl_codebook)

            # 2.2 - Initialize dictionary
            VPL_dic    = {}
            user_ids   = []
            for field in codebook.keys():
                field_common = codebook[field]['field_common']
                in_dico      = codebook[field]['in_dico']
                if in_dico == '1':
                    VPL_dic[field_common] = []

            # 2.3 - Open VPL File
            cr_VPL_user = csv.reader(open(vpl_file_user,"rU"))
            VPL_header  = cr_VPL_user.next()
            twitter_idx = VPL_header.index('Twitter_ID')

            # 2.4 - Fill in dictionary
            Twitter_IDs = []
            for user in cr_VPL_user:
                Twitter_ID = str(user[twitter_idx])
                # Check if user has a Twitter information
                if Twitter_ID != 'NA':
                    Twitter_IDs.append(Twitter_ID)
                    # Fill in dictionary
                    for field in codebook.keys():
                        field_common = codebook[field]['field_common']
                        in_dico      = codebook[field]['in_dico']
                        if in_dico == '1':
                            VPL_dic[field_common].append(user[VPL_header.index(field)])

            # 2.5 Filter VPL_dict entries
            VPL_dics = []
            VPL_dics.append(filter_entries(VPL_dic.copy(),codebook,chosen_parties))
            VPL_dics.append(filter_entries(VPL_dic.copy(),codebook,major_parties))

            if task == 'ML':
                VPL_dic = filter_entries(VPL_dic,codebook,major_parties)
            elif task == 'DESC' or task == 'FILTERS':
                VPL_dic = filter_entries(VPL_dic,codebook,chosen_parties)

            # test = np.asarray(VPL_dic['vote_intention'])
            # test = test[~np.isnan(test)]
            # print(np.unique(test))
            #print(np.unique(VPL_dics[1]['vote_intention']))
            #print(chosen_parties)
            #print(major_parties)
            #exit()

            # 3) VPL_dic info to the user_dic
            cnt=  0
            for field in VPL_dic:

                # Add in dic
                Twitter_IDs_copied = list(Twitter_IDs) # to avoid pythonic pointer problem
                add_in_dic(user_dic,Twitter_IDs_copied,list(VPL_dic[field]),field)
                del Twitter_IDs_copied # Suppress the intermediate variable
                cnt = cnt + 1
                if cnt % every_n == 0:
                    print('Number of user fields handled:'+str(cnt))
            del VPL_dic

            # 4) Add colors to the user dic
            print('Add colors')
            (keys,matrix)       = extract(user_dic,color_field)
            Twitter_IDs_copied  = list(keys)
            colors_field_info   = list(matrix[:,0])
            colors              = [hex2int(colors_parties[int(x)]) for x in colors_field_info]
            add_in_dic(user_dic,Twitter_IDs_copied,colors,'colors')

            # Step 6 : Create dependent fields
            create_dependent_fields(user_dic,'questions',nquestions)
            #print user_dic[user_dic.keys()[0]]
            # Step 7 :  Save dictionary
            pickle.dump(user_dic, open(user_dic_file, "wb" ) )
            print('User dictionary written with success')
