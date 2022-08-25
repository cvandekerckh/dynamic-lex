#-*- coding:utf-8 -*-
#!/usr/bin/env python2.7
# ------------------------------------------------------------------------------
# Filename:    1_create_matrix.py
# Description: Creates a sparse network matrix (nxm; n = users, m = candidates)
# ------------------------------------------------------------------------------
import csv
from appendix import return_election,create_data_struct

elections = return_election()

print "Creating data structure"
create_data_struct('../', 'data_struct.txt', elections)


# 2 : Create network matrix
for election in elections:

    # Input
    network_file     = '../data/raw/follow/'+election+'/network.csv'
    politicians_file = '../data/raw/follow/'+election+'/politicians_link.csv'
    # network.csv contains each candidate id matched with their followers (vpl data)
    # politicians_link.csv general info on the Twitter IDs (eg.account name)

    # Output for sparse matrix
    clean_file              = '../data/clean/'+election+'/sparse_matrix.csv'

    # Declare sparse lists
    i_idx         = list() # row 1 : i index
    j_idx         = list() # row 2 : j index
    i_names       = list() # row 3 : i name
    j_names       = list() # row 4 : j name
    j_screennames = list() # row 4 : j screenname

    # Compute network size
    with open(network_file,'rb') as f:
        for n_rows, lines in enumerate(f):
                pass

    # Open files
    cr_net = csv.reader(open(network_file,'rb'))
    cr_pol = csv.reader(open(politicians_file,'rU'))

    pol_header = cr_pol.next()
    name_idx   = pol_header.index('Twitter_account')
    id_idx     = pol_header.index('Twitter_ID')

    # Create politician list
    pol_list = list()
    for politician in cr_pol:
        j_screennames.append(politician[name_idx])
        j_names.append(politician[id_idx])

    # Count the number of VPL followers for each politician
    pol_ctr = [0]*len(j_names)

    # Fill sparse lists
    i = 1
    cr_net.next() # pass header
    for user in cr_net:
        i_names.append(user[0])
        for id_n in user:
            if id_n in j_names:
                i_idx.append(i)
                j = j_names.index(id_n)
                j_idx.append(j+1) # in R, indices starts at 1
                pol_ctr[j] = pol_ctr[j] + 1
        if i%1000 == 0:
            print('progression : %d / %d users ' % (i,n_rows))
        i = i+1



    # Print in output file
    print('Number of entries in matrix: '+str(len(i_idx))+'/'+str(i*len(j_names)))

    # Write in output files
    with open(clean_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(i_idx)
        writer.writerow(j_idx)
        writer.writerow(i_names)
        writer.writerow(j_names)

    print('Sparse matrix written \n')
