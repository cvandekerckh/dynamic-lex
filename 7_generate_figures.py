#-*- coding:utf-8 -*-
#!/usr/bin/env python2.7
# ------------------------------------------------------------------------------
# Filename:    7_generate_figures.py
# Description: Makes figures from dictionary files
# ------------------------------------------------------------------------------

import cPickle as pickle
import csv
import math
import matplotlib
import matplotlib.patches as patches
import matplotlib.path as path
import matplotlib.pyplot as plt
import numpy as np
import random as rd
import re
from appendix import extract, file_to_dic, write_interesting, int2hex, add_plot, plot_matrix
from appendix import select_features, colors_for_plot, plot_kiviat
from scipy.stats import pearsonr, spearmanr
from sklearn import svm, datasets
from sklearn.ensemble import ExtraTreesClassifier
from appendix import return_election,read_election
from sklearn.metrics import average_precision_score, precision_recall_curve

##########################################################################
############################ 0 - RAW INPUT ###############################
##########################################################################

print '(1) Fig. 1 - politicians ideologies \n(2) Fig. 2 - citizens ideologies   \n(3) Fig. 3 - Vote pred. (global) \n(4) Fig. 4 - Vote pred. (per party) \n(5) Tab. 1 - citizens ideologies (detailed) \n(6) Appendix Figure 1 \n(7) Appendix Figure 4 \n(8) Appendix Figure 2 \n(9) Appendix Figure 3'

data_folder = 'data'
elections = return_election()

for plot_state in range(1,10):
    if plot_state in [1,2,5,8]: # Non machine learning figures

        for election in return_election():

            # Parameters
            follow_type = 'follow_nuts_party'
            text_type   = 'text_wf_2gram'
            print_corr_matrix = False
            write_data = False
            fig_size = (9,8)

            # Input
            path     = '../'+data_folder+'/dictionaries/'+election+'/'

            # Open dictionaries
            candidate_dic            = pickle.load(open(path+'candidate_dic.p',"rb"))
            candidate_to_display_dic = pickle.load(open(path+
                                       'candidate_to_display_dic.p',"rb"))
            user_dic      = pickle.load(open(path+'user_dic.p',"rb"))
            vpl_file = '../'+data_folder+'/raw/vpl'+election+'-politicians-extended.csv'

            # Output
            plot_info = [1,1] # Subplot disposition

            # Politicians
            if plot_state == 1:
                fig = plt.figure(figsize=fig_size)
                title = 'ELITE-FOLLOW-TEXT'

                # Extract matrix and filter outliers
                fields        = [follow_type,text_type,'colors']
                (keys,matrix) = extract(candidate_dic,fields)

                # Plot
                to_extract = [follow_type,text_type,'colors']
                add_plot(candidate_dic,to_extract,title = None,scale = True,xlabl = 'Network Ideology',ylabl = 'Textual Ideology', filter_outliers = True)
                plt.savefig('../data/output/%s/Figure_1.pdf' % election)


            # Users
            elif plot_state == 5:
                cw = csv.writer(open('../data/output/%s/Figure_5.csv' % election,'w'))
                cw.writerow(['Method','Quality'])
                estimates =  [follow_type,'text_wf_2gram','text_wgt_2gram','text_wgt2_2gram']
                estimates_names = ['network','textual_baseline1','textual_weighted','textual_new']
                for mm,estimate in enumerate(estimates):
                    common_fields = [estimate,'questions','colors']
                    if estimate in user_dic[user_dic.keys()[0]]:
                        (keys,matrix) = extract(user_dic,common_fields)
                        print('Pearson correlation '+str(estimate)+' - VPL :')
                        print(np.corrcoef(matrix[:,0],matrix[:,1])[0,1])
                        cw.writerow([estimates_names[mm],abs(np.corrcoef(matrix[:,0],matrix[:,1])[0,1])])

            elif plot_state == 2:
                fig = plt.figure(figsize=fig_size)
                title = 'USER-MIX'

                (keys,matrix) = extract(user_dic,[follow_type,'text_wgt2_2gram','questions','colors'])
                n_res = 100
                lambdas = np.linspace(0,1,n_res)
                correlations = np.zeros(n_res)
                i = 0
                for lmbda in lambdas:
                    if election == '2015-can-canada-sample':
                        inv = 0
                    else:
                        inv = 1
                    ideologies = lmbda*(pow(-1,inv)*matrix[:,0]) + (1-lmbda)*(matrix[:,1])
                    correlations[i] = abs(np.corrcoef(ideologies,matrix[:,2])[0][1])
                    i = i + 1

                # Print the max value
                max_lambda = lambdas[0]
                max_corr   = correlations[0]
                LL         = len(lambdas)
                for i in range(len(correlations)):
                    if correlations[i] > max_corr:
                        max_lambda = lambdas[i]
                        max_corr   = correlations[i]
                print('Max lambda: '+str(max_lambda))
                print('Max corr:'+str(max_corr))
                print('Network corr:'+str(correlations[len(correlations)-1]))
                plt.plot(lambdas,correlations,linewidth=6.0)
                plt.plot([0,1],[max_corr,correlations[LL-1]],ls = 'dashed',
                linewidth = 6.0, color = 'grey')
                plt.xticks(color='k', size=38)
                plt.yticks(color='k', size=38)
                plt.xlabel('$\lambda$',labelpad=10, size = 60)
                plt.ylabel('Correlation ',labelpad=15 , size = 50)

                plt.ylim([0,1])

                ax = plt.gca()
                ax.tick_params(axis='both', which='major', pad=15)
                ax.annotate('$\hat{\phi}_{txt}$', xy=(0, 100*correlations[0]), xytext=(0.03, 100*correlations[0]-14),fontsize = 60)
                ax.annotate('$\hat{\phi}_{net}$', xy=(1, 100*correlations[LL-1]), xytext=(0.75, 100*correlations[LL-1]+7),fontsize = 60)
                fig.set_tight_layout(True)
                plt.savefig('../data/output/%s/Figure_2.pdf' % election)

            elif plot_state == 8:

                # 1 - Choose range per election
                k_limit = 23 # Arbitrary range for plot
                if election == '2015-can-canada':
                    k_limit = 2*k_limit # offer longer period for bigger election

                # 2) Extract feature generated by multiple filter
                path          = '../data/filters/dictionaries/%s/' % election
                candidate_dic = pickle.load(open(path+'candidate_dic.p',"rb"))
                features = candidate_dic[candidate_dic.keys()[0]].keys()
                multiple_tuples  = []
                kept_feature     = []
                id_feature       = []

                for feature in features:
                    multiple_feature = re.search(r'wf_2gram_(\d+)',feature)

                    if multiple_feature is not None:
                        new_k = int(multiple_feature.group(1))
                        if new_k <= k_limit:
                            kept_feature.append(feature)
                            id_feature.append(new_k)

                (keys,matrix) = extract(candidate_dic,['follow_nuts_party']+kept_feature)
                n_cnd         = len(extract(candidate_dic,['follow_nuts_party'])[0])

                # 3 - Compute correlations

                for k in id_feature:
                    corr = pearsonr(matrix[:,0],matrix[:,id_feature.index(k)+1])
                    if corr[1] <= .05:
                        multiple_tuples.append((k,(corr[0])))

                # 4 - Plot curve
                multiple_tuples = sorted(multiple_tuples)

                X = [100*item[0]/float(n_cnd) for item in multiple_tuples]
                Y = [abs(item[1]) for item in multiple_tuples]


                fig = plt.figure()
                ax  = plt.subplot(111)
                plt.plot(X,Y,'r-',lw = 4)
                plt.xticks(size=25)
                plt.yticks(size=25)
                plt.title('Politician word filter - %s' % election)

                #plt.yticks(np.linspace(30,100,8), color='k', size=25)
                plt.yticks(np.linspace(0.3,1,8), color='k', size=25) # @coco : review

                plt.xlabel(r'Filter threshold $\beta$',{'fontsize':25})
                #ax.xaxis.labelpad = 20
                #ax.yaxis.labelpad = 20
                ax.tick_params(pad=8)
                #plt.ylabel('Correlation [%]',{'fontsize':25})#,'color':'white'})
                plt.ylabel('Correlation ',{'fontsize':25})#,'color':'white'})

                #plt.title(election,fontsize=18)
                #plt.legend(loc="lower right",fontsize = 20)
                fig.set_tight_layout(True)
                plt.savefig('../data/output/%s/Appendix_2.pdf' % election)

    elif plot_state in [3,4,7]:

        elections = ['2015-can-canada','2014-nzl-new-zealand','2014-can-quebec']

        # Path and parameters
        path      = '../'+data_folder+'/machine_learning/'
        target    = 'users'
        prefixes  = ['C','N','Q']
        el_colors = ['#E38E00','black','blue']

        # Choose features present in Figure
        if plot_state == 3:
            sources   = [
            'follow','content','SP-questions',
            'follow-content','content-SP-questions','follow-SP-questions',
            'follow-content-SP-questions']
            source_names = ['network','text','survey',
            'network + text','text + survey','network + survey',
            'network + text + survey']
        elif plot_state == 4:
            sources    = ['random'      ,'follow-content',
                          'SP-questions','real_random']
        elif plot_state == 7:
            sources   = ['random','follow','content','SP-questions','real_random']
            source_names = {'follow':'Network','content':'Text','follow-content':'Twitter',
                             'SP':'Self','questions':'Questions','SP-questions':'Survey'}
            source_colors = {'follow':'#0040ff','content':'#00bfff','follow-content':'#0000ff',
                             'SP':'#ffbf00','questions':'#ff4000','SP-questions':'#ff0000',
                             'real_random':'grey'}
            thres_dic = {'2015-can-canada':.008,'2014-nzl-new-zealand':.008,'2014-can-quebec':.015} # dictionary to improve the plotting display

        # Plot Options : Slope graph
        slope_source     = ['SP-questions','follow-content']
        slope_sourcename = ['Survey','Twitter']
        slope_colors     = {1:'g',-1:'r--',0:'g'}

        n_info = 2 # Number of info lines printed in header

        if plot_state == 4:
            fig = plt.figure(figsize=(8,8))
        elif plot_state == 7:
            fig, axes = plt.subplots(ncols=len(elections), sharey=True, figsize = (15,10))

        for election in elections:
            if plot_state == 3:
                print('\n ***'+election+'***')
                print('Venn diagram - Performances Follow-Text-Survey')
                cw = csv.writer(open('../data/output/%s/Figure_3.csv' % election,'w'))
                cw.writerow(['AUC','features'])
            slope_flag = 0 # Flag that is activated at each new election
            [abrev_parties,colors_parties] = read_election(election,['party_abr','party_colors'])
            abrev_parties = abrev_parties.split(',')
            colors_parties = colors_parties.split(',')
            cnt = 0


            for kk,source in enumerate(sources):
                filename = path+election+'_'+source+'_'+target

                # Election properties
                cr_valid = csv.reader(open(filename+'_y-valid.csv','r'))
                for i in range(n_info):
                    cr_valid.next()
                header   = cr_valid.next()
                M         = sum(1 for row in cr_valid)
                n_classes = len(header)

                # Create a slope dictionary
                if plot_state == 4 and slope_flag == 0:
                    slope_flag = 1
                    slope_dic  = dict.fromkeys(header)
                    for key in slope_dic:
                        slope_dic[key] = [0,0]

                # Create y_score and y_valid
                cr_valid = csv.reader(open(filename+'_y-valid.csv','r'))
                cr_score = csv.reader(open(filename+'_y-score.csv','r'))
                for i in range(n_info+1):
                    cr_valid.next()
                    cr_score.next()
                y_score = np.NAN*np.zeros((M,n_classes))
                y_valid = np.NAN*np.zeros((M,n_classes))
                i = 0
                for line in cr_valid:
                    y_valid[i,:] = line
                    i = i + 1
                i = 0
                for line in cr_score:
                    y_score[i,:] = line
                    i = i + 1

                # Compute Precision-Recall and plot curve
                precision = dict()
                recall = dict()
                average_precision = dict()
                for i in range(n_classes):
                    precision[i], recall[i], _ = precision_recall_curve(y_valid[:, i],
                                                                        y_score[:, i])
                    average_precision[i] = average_precision_score(y_valid[:, i], y_score[:, i])

                # Compute micro-average ROC curve and ROC area
                precision["micro"], recall["micro"], _ = precision_recall_curve(y_valid.ravel(),
                    y_score.ravel())
                average_precision["micro"] = average_precision_score(y_valid, y_score,average="micro")

                if plot_state == 3:
                    AUC = round(100*average_precision["micro"],2)
                    cw.writerow([AUC,source_names[kk]])
                    if len(source.split('-')) > 1:
                        print 'AUC = %s when using %s ' % (str(AUC),source)
                    else:
                        print 'AUC = %s when using only %s ' % (str(AUC),source)

                    continue

                elif plot_state == 7:
                    # Arrange view problems for low recalls
                    thres = thres_dic[election]
                    recall["micro"][recall["micro"] < thres] = np.NAN
                    if sum(recall["micro"] < .2) < 1: # If no small recall is recorded
                        idx = np.isnan(recall["micro"]).tolist().index(1) # Find first nan
                        recall["micro"][idx] = thres
                        precision["micro"][idx] = precision["micro"][idx-1]


                ############################################
                ################# PLOTTING #################
                ############################################

                if plot_state == 7:
                    if len(elections) > 1:
                        ax = axes[elections.index(election)]
                    else:
                        ax = axes

                    # Grid design
                    ax.grid(zorder=0)
                    ticklines = ax.get_xticklines() + ax.get_yticklines()
                    gridlines = ax.get_xgridlines() + ax.get_ygridlines()

                    for line in ticklines:
                        line.set_linewidth(10)

                    for line in gridlines:
                        line.set_linestyle('-')
                        line.set_linewidth(0.5)
                        line.set_color([0.6,0.6,0.6])
                else:
                    ax = plt.subplot(111)

                if plot_state == 4:
                    for i in range(n_classes):
                        class_name = header[i]
                        if source in slope_source:
                            AUC = 100*average_precision[i]
                            slope_dic[class_name][slope_source.index(source)] = AUC

                # 3) Plot micro graph
                if plot_state == 7:
                    if source == 'random':
                        precision_middle = len(precision)/2
                        ax.axhspan(-0.5, precision["micro"][precision_middle], color='grey', alpha=0.5, lw=0)
                    else:
                        if source == 'real_random':
                            ls = '--'
                        else:
                            ls = '-'
                            ax.annotate(source_names[source]+' : '+str(round(average_precision["micro"],2)), xy=(0.0 + cnt/4.0, np.max(precision["micro"]-0.1)), xytext=(0.0 + cnt/4.0, np.max(precision["micro"]-0.3-cnt/4.0)),
                            color=source_colors[source], fontsize=20)
                            cnt = cnt + 1

                        ax.plot(recall["micro"], precision["micro"], zorder = 3, color = source_colors[source],
                             linestyle = ls,label='{0} (area = {1:0.2f})'
                                   ''.format(source,average_precision["micro"]))

            # 3bis) Plot Slope graph per party
            if plot_state == 4:
                eps_1 = 0.28
                eps_2 = 0.1
                head_eps = 10
                head_h = 0.16 # horizontal title
                head_p = 0.4 # horizontal parties
                for key in slope_dic:
                    [v1,v2] = slope_dic[key]
                    plt.plot([1,2],[v1,v2],slope_colors[np.sign(v2-v1)])
                    #marker = '.',      # obsolete
                    #markersize = '20')

                    # Header
                    ax.annotate('Survey',xy=(1-head_h,100+head_eps),xytext=(1-head_h,100+head_eps),fontsize = 25)
                    ax.annotate('Twitter',xy=(2-head_h,100+head_eps),xytext=(2-head_h,100+head_eps),fontsize = 25)
                    ax.annotate('Parties',xy=(2+head_p,100+head_eps),xytext=(2+head_p,100+head_eps),fontsize = 22)

                    # Right
                    prefix = prefixes[elections.index(election)]
                    el_col = el_colors[elections.index(election)]
                    ax.annotate(round(v2,1), xy=(2,v2), xytext=(2+eps_2,v2), fontsize=20)
                    ax.annotate(prefix+'-'+key, xy=(2,v2), xytext=(2+4*eps_2,v2), fontsize=20,color = el_col)
                    # Left
                    ax.annotate(round(v1,1), xy=(1,v1), xytext=(1-eps_1,v1), fontsize=20)

            # General plot design
            if plot_state == 7:
                ax.set_xlim([0, 1.0])
                ax.set_ylim([-0.06, 1.35])
                #plt.plot([0,1],[1.0/n_classes,1.0/n_classes], 'k--',lw = 2,)
                #ax.set_xticks(np.linspace(0,1,6), size=25)
                ax.xaxis.set_tick_params(labelsize=20)
                #ax.set_yticks(np.linspace(0,1,6), size=25)
                ax.yaxis.set_tick_params(labelsize=20)
                ax.set_xlabel('Recall',{'fontsize':25})
                ax.xaxis.labelpad = 20
                ax.yaxis.labelpad = 20
                ax.tick_params(axis='both', which='major', pad=8)
                ax.set_ylabel('Precision',{'fontsize':25})


                #plt.title(election,fontsize=18)
                #plt.legend(loc="lower right",fontsize = 20)
                fig.set_tight_layout(True)
                plt.savefig('../data/output/Appendix_4.pdf')

        if plot_state == 4:
           eps_x = 0.4
           eps_y = 20
           plt.xlim([1-eps_x,2+4*eps_2+eps_x])
           plt.ylim([20, 100+eps_y])
           plt.xticks([1,2],['',''])

           plt.yticks(size=0)
           ax.xaxis.labelpad = 10
           ax.yaxis.labelpad = 10
           plt.ylabel('Average Prediction Efficiency',{'fontsize':25})
           ax.tick_params(axis='both', which='major', pad=11)
           ax.yaxis.grid(False)
           ax.xaxis.grid(False)

           fig.set_tight_layout(True)
           plt.savefig('../data/output/Figure_4.pdf')

    # Kiviat plot
    elif plot_state == 6:
        # Step 1: Define parameters
        CORR  = 'Quality'
        COMPL = 'Classification'
        RANG  = 'Information'
        elections  = ['2015-can-canada','2014-can-quebec','2014-nzl-new-zealand']

        properties = [CORR,COMPL,RANG]
        grams      = [1,2,3]

        # Step 2: Fill the gram dictionary
        gram_dic = dict.fromkeys(grams)
        for key in gram_dic:
            gram_dic[key] = dict.fromkeys(properties)
            for prop in properties:
                gram_dic[key][prop] = []

        for election in elections:
            # a) Load dictionary
            path     = '../'+data_folder+'/dictionaries/'+election+'/'
            #path          = '../../data/output/dictionaries'+final_fix+'/'+election+'/'
            candidate_dic = pickle.load(open(path+'candidate_dic.p',"rb"))

            # b) Extract a common dataset extraction
            follow_type = 'follow_nuts_party'
            text_type   = ['text_wf_'+str(x)+'gram' for x in grams]

            common_fields = [follow_type] + text_type + ['colors']
            (keys,matrix) = extract(candidate_dic,common_fields)

            n_min = 40 # Minimal sample to consider a correlation

            # c) Fill in gram dic with properties
            for gram in gram_dic:
                for prop in properties:
                    if prop == CORR:
                        if(len(keys) > n_min):
                            gram_dic[gram][prop].append(abs(pearsonr(matrix[:,common_fields.index(follow_type)],matrix[:,gram])[0]))
                    if prop == RANG:
                        Nin  = len(extract(candidate_dic,[follow_type,'colors'])[0])
                        Nout = len(extract(candidate_dic,[follow_type,'text_wf_'+str(gram)+'gram','colors'])[0])
                        gram_dic[gram][prop].append(Nout/float(Nin))
                    if prop == COMPL:
                        # a) Classification with only follow
                        y   =  matrix[:,common_fields.index('colors')]
                        idx = [common_fields.index(follow_type)]
                        X   =  matrix[:,idx]
                        svc = svm.LinearSVC(C=1,random_state = 0).fit(X, y)
                        score_1 = svc.score(X, y)

                        # b) Classification with follow and text
                        y =  matrix[:,common_fields.index('colors')]
                        idx = [common_fields.index(follow_type),gram]
                        X =  matrix[:,idx]
                        svc = svm.LinearSVC(C=1,random_state = 0).fit(X, y)
                        score_2 = svc.score(X, y)

                        if score_1 != 1 and (score_2 > score_1): # avoid division by 0 if no improvement to make
                            improvmt = (score_2 - score_1)/(1-score_1)
                            gram_dic[gram][prop].append(improvmt)
                        else:
                            pass

        # d) Take the average of each list
        for gram in gram_dic:
            for prop in gram_dic[gram]:
                gram_list = gram_dic[gram][prop]
                gram_dic[gram][prop] = np.mean(gram_list)

        ###### PLOTTING PART
        # Plot the radar diagram

        matplotlib.rc('axes', facecolor = 'white')
        fig = plt.figure(figsize=(17,4), facecolor='white')
        for gram in grams:
            plt.subplot(1,len(grams),gram)
            values = [100*gram_dic[gram][x] for x in properties]
            plot_kiviat(properties,values,gram,len(grams),-np.pi/6)

        plt.subplots_adjust(wspace=.9)
        plt.savefig('../data/output/Appendix_1.pdf')

    # Display user filter
    elif plot_state == 9:
        filter_conds = [x for x in range(2,36)]

        elections = ['2015-can-canada','2014-nzl-new-zealand','2014-can-quebec']
        elections_abrev = ['CAN','NZL','QUE']
        # Dictionary to plot
        dic_9 = {election:[[],[]] for election in elections}

        for election in elections:
            path          = '../data/filters/dictionaries/%s/' % election
            user_dic = pickle.load(open(path+'user_dic.p',"rb"))
            follow_type = 'follow_nuts_party'

            for filter_cond in filter_conds:
                fields = [follow_type,'text_wgt2_2gram_'+str(filter_cond),'questions','colors']
                (keys,matrix) = extract(user_dic,fields)
                x1 = matrix[:,1]
                x2 = matrix[:,2]
                rho = abs(np.corrcoef(x1,x2)[0,1])
                dic_9[election][0].append(rho)
                dic_9[election][1].append(len(keys))

        # Plot graph
        fig = plt.figure(figsize = (10,5))

        default_filter = 25 # default value for filter
        ref_idx     = filter_conds.index(default_filter)
        colors = ['r','#2d69a9','g']
        styles = ['-','--','--']
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212)
        legend_plts = []
        for jj,election in enumerate(elections):
            legend_plts.append(ax1.plot(filter_conds,dic_9[election][0], c= colors[jj],linestyle = styles[jj], linewidth = 3)[0])
            sample_perc = 100* np.asarray(dic_9[election][1])/dic_9[election][1][ref_idx]
            ax2.plot(filter_conds,sample_perc, c = colors[jj], linestyle = styles[jj],linewidth = 3)
        ax1.grid(linestyle='-')
        ax1.yaxis.set_ticks([0.2,0.3,0.4,0.5,0.6])
        ax1.set_title('Correlation')
        ax2.grid(linestyle='-')
        ax2.yaxis.set_ticks([100,300,500,700])
        ax2.set_title('Sample variation (%)')
        ax1.plot([int(default_filter),int(default_filter)],[0.1,0.6],c = 'black',linewidth = 1.5)
        ax2.plot([int(default_filter),int(default_filter)],[50,750],c = 'black',linewidth = 1.5)
        plt.tight_layout()
        ax1.legend(legend_plts,elections_abrev)
        ax2.legend(legend_plts,elections_abrev)
        plt.savefig('../data/output/Appendix_3.pdf')

    else:
        print('Wrong input (%s)' % str(plot_state))
# Print end msg
print('Function ended with success')
#plt.show()
