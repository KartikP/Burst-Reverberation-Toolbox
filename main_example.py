import pandas as pd
from organize_spikelist import organize_spikelist
import run_brt
import matplotlib.pyplot as plt
import numpy as np
import detect_reverb_clusters
import seaborn as sns

filename = 'spikelist.csv'
spikelist = pd.read_csv(filename, error_bad_lines=False, low_memory=False)
spikelist = organize_spikelist(spikelist)  # Required if spike list is raw/unedited
# For SHANK2 dataset, you will need to change the dataframe column name to match Rett spiketime csv

'''
Change Well (A1 -> C4)
'''
well = 'C3'

well_spiketimes = spikelist[spikelist['Electrode'].str.contains(well)]

well_reverb = run_brt.main(well_spiketimes)

burst_peaks = well_reverb[2]
sb_start = well_reverb[6]
sb_end = well_reverb[7]
weighted_network_sdf = well_reverb[-2]
raster = well_reverb[-1]
burst_borders = well_reverb[4]

isReverb, percent_overlapping_x, percent_overlapping_y, x, y, labels, centers, dst = detect_reverb_clusters.k_means(burst_peaks, burst_borders, weighted_network_sdf)

'''
Plot Raster, SDF, and burst borders
'''
fs=12500
plt.figure(figsize=(9, 6))
plt.eventplot(raster, color='black', linelengths=0.5, linewidths=0.75, alpha=0.35)
plt.plot(np.arange(0,300,(1/fs)), weighted_network_sdf, 'black')
plt.plot(burst_peaks, weighted_network_sdf[(burst_peaks*fs).astype(int)],'.', color='red', alpha=0.65)
for i in range(len(sb_start)):
    plt.axvline(sb_start[i], color='green', linewidth='1', alpha=0.35)
    plt.axvline(sb_end[i], color='red', linewidth='1', alpha=0.35)

'''
Plot Clustering
'''
palette = sns.color_palette("rocket")
plt.figure(figsize=(5, 5))
sns.scatterplot(x=x, y=y, palette='rocket', hue=labels)
plt.xlabel("IBI (s, Standardized)")
plt.ylabel("FR at peak (Hz, Standardized)")
plt.legend([], [], frameon=False)

