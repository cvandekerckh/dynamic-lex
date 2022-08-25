import os
import pandas as pd
import numpy as np
from appendix import return_election,create_data_struct
from shutil import copyfile
import csv

np.random.seed(seed=0)
elections = return_election()
# 0 ) Create data structure
create_data_struct('../', 'data_struct.txt', elections)

# 1) Create a list with all the potential twitter ids
directories = ['features_desc','features_ml']


for election in elections:

    ############## VPL FOLDER ####################
    print('1) VPL')
    target_ids = []
    for directory in directories:
        for file in os.listdir('../%s/%s/%s/' % ('data_ext',directory,election)):
            if file.endswith(".csv"):
                df = pd.read_csv('../%s/%s/%s/%s' % ('data_ext',directory,election,file))
                new_ids = df['key'].tolist()
                target_ids = target_ids + list(set(new_ids) - set(target_ids))
                #print('- added %s (total: %d): ' % (file,len(target_ids)))
    target_ids = np.unique(target_ids)
    print('\nInteresting sample for %s : %d' % (election,len(target_ids)))

    # 2) Generate random sample with with priority to the interesting sample
    sample_size = 5000
    df = pd.read_csv('../data/raw/vpl/%s-extended.csv' % election,na_values = 'NA')
    print('Original df size : (%d,%d)'% df.shape)
    df = df.drop_duplicates(subset='Twitter_ID', keep="first")
    df = df[df['Twitter_ID'].isin(df['Twitter_ID'].dropna())]

    all_ids = df['Twitter_ID'].tolist()
    all_ids = np.unique(all_ids)
    print('Total sample for %s : %d' % (election,len(all_ids)))

    # Delete twitter ids not in extended
    strange_ids = np.setdiff1d(target_ids,all_ids,assume_unique = True)
    print('Total of uncoherent ids : %s' % len(strange_ids))
    target_ids  = [item for item in target_ids if item not in strange_ids]

    # Generate the random sample
    remaining_ids = np.setdiff1d(all_ids,target_ids,assume_unique = True).tolist()
    ordered_ids   = np.random.permutation(target_ids).tolist() +          np.random.permutation(remaining_ids).tolist()
    chosen_ids    = ordered_ids[0:sample_size]
    df = df[df['Twitter_ID'].isin(chosen_ids)]
    print('Final df size : (%d,%d)'% df.shape)
    df.to_csv('../data/raw_filtered/vpl/%s-extended.csv' % (election),na_rep = 'NA',index = False)

    # Simple copy of the other files
    src = '../data/raw'
    dst = '../data/raw_filtered'
    copyfile('%s/vpl/%s-politicians-extended.csv' % (src,election),    '%s/vpl/%s-politicians-extended.csv' % (dst,election))
    copyfile('%s/vpl/%s-codebook.csv' % (src,election), '%s/vpl/%s-codebook.csv' % (dst,election))

    ############ FOLLOW FOLDER #########
    print('2) FOLLOW')

    copyfile('%s/follow/%s/politicians_init.csv' % (src,election),    '%s/follow/%s/politicians_init.csv' % (dst,election))
    copyfile('%s/follow/%s/politicians_link.csv' % (src,election), '%s/follow/%s/politicians_link.csv' % (dst,election))

    chosen_ids = [str(x) for x in chosen_ids]
    ids = []
    with open('%s/follow/%s/network.csv' % (src,election),'r') as fin:
        with open('%s/follow/%s/network.csv' % (dst,election),'w') as fout:
            cr = csv.reader(fin)
            cw = csv.writer(fout)
            cw.writerow(cr.next())
            cnts = [0,0]

            for line in cr:
                cnts[0] += 1
                if (line[0] in chosen_ids) and (line[0] not in ids):
                    ids.append(line[0])
                    cw.writerow(line)
                    cnts[1] += 1
            print('Number of lines written in follow : (%d/%d)' % (cnts[1],cnts[0]))

    ############## TXT FOLDER ####################
    print('3) TEXT')

    for file in os.listdir('%s/text/%s/' % (src,election)):
        if file.startswith('politicians'):
            copyfile('%s/text/%s/%s' % (src,election,file),
            '%s/text/%s/%s' % (dst,election,file))

    # Handle link and TM2gram
    df_link = pd.read_csv('%s/text/%s/link.csv' % (src,election))
    df_link['Twitter_ID'] = [str(x) for x in df_link['Twitter_ID']]
    text_ids = df_link['Twitter_ID'].tolist()
    hdr = ['words'] + text_ids
    df_tm   = pd.read_csv('%s/text/%s/TM2gram.csv' % (src,election),sep = ' ',header = None, names = hdr)
    old_shape = df_tm.shape

    # Filter TM
    text_chosen_ids = [x for x in chosen_ids if x in text_ids]
    df_tm   = df_tm[['words'] + text_chosen_ids]
    _, i = np.unique(df_tm.columns, return_index=True)
    df_tm = df_tm.iloc[:, np.sort(i)]

    # Recreate a link file from tm
    df_link = df_link.drop_duplicates(subset = 'Twitter_ID')
    df_link_new = pd.DataFrame(df_tm.columns[1:], columns=['Twitter_ID'])
    df_link_new = pd.merge(df_link_new, df_link, on = 'Twitter_ID',how = 'left')

    print('Old shape (%d,%d)--> New shape : (%d,%d)' % (old_shape[0],old_shape[1],df_tm.shape[0],df_tm.shape[1]))
    print('link file : %d rows' % df_link_new.shape[0])

    df_tm.to_csv('%s/text/%s/TM2gram.csv' % (dst,election),header = False,index = False,sep = ' ')
    df_link_new.to_csv('%s/text/%s/link.csv' % (dst,election),index = False)
