import pandas as pd
'''
Reorganizes the Axion spikelist into a dataframe
'''
def organize_spikelist(spikelist):
    '''
    Remove unwanted columns and rows
    '''
    spikelist.drop(['Investigator', 'Unnamed: 1', 'Amplitude(mV)'], axis=1, inplace=True)
    spikelist = spikelist[:-1]

    return spikelist
