import burst_reverberation_toolbox as brt
import numpy as np
import pandas as pd
import detect_reverb_clusters as drc

'''
To do:
Implement dynamic threshold using ISI distribution!
'''

def main(well_spiketimes):
    fs = 12500
    raster = []
    sdf = []
    maxFR = []
    channel_id = []
    potential_burst_peaks = []
    partial_burst_peaks = []
    gaussian_kernel = brt.generate_gaussian_kernel(sigma=0.075)

    # Store values
    prime_burst_peaks = np.nan
    burst_peaks = np.nan
    partial_burst_peaks = np.nan
    burst_borders = np.nan
    rmax = np.nan
    sb_start = np.nan
    sb_end = np.nan
    num_reverbs = np.nan
    iri = np.nan
    sb_duration = np.nan
    isbi = np.nan
    fraction_of_participating_electrodes = np.nan
    burst_strength = np.nan
    burst_window = np.nan
    adaptation = np.nan
    synchrony = np.nan
    STTC_matrix = np.nan
    isReverb = False

    # Identify channels to organize
    channels_in_well = sorted(pd.unique(well_spiketimes['Electrode']))

    for channel in channels_in_well:
        channel_id.append(channel)
        spiketrain = well_spiketimes[well_spiketimes['Electrode'] == channel]['Time (s)']
        raster.append(spiketrain)
        spikematrix = brt.generate_spikematrix(spiketrain)
        sdf_tmp = brt.generate_sdf(spikematrix, gaussian_kernel)
        sdf.append(sdf_tmp)
        maxFR.append(max(sdf_tmp))
    network_sdf = np.mean(sdf, axis=0)
    weighted_network_sdf = np.average(sdf, axis=0, weights=maxFR)

    #STTC_matrix = brt.sttc(raster, channel_id)

    # Check if network is bursting
    if max(weighted_network_sdf) < 5:
        print(f"Network is not bursting.")
    else:
        '''
        The follow 10-15 lines of code are for detecting "burst peaks", and filtering out noise peaks (the latter 
        doesn't work too well).
        '''
        prime_burst_threshold = np.max(weighted_network_sdf) * 0.4
        prime_burst_peaks = brt.detect_burst_peaks(weighted_network_sdf, prime_burst_threshold)

        potential_burst_threshold = np.std(weighted_network_sdf) * 0.01
        if potential_burst_threshold < 0.2:
            potential_burst_threshold = 0.2
        potential_burst_peaks = brt.detect_burst_peaks(weighted_network_sdf, potential_burst_threshold)

        burst_threshold = np.std(potential_burst_peaks) * 0.01
        burst_peaks = potential_burst_peaks[weighted_network_sdf[(potential_burst_peaks*fs).astype(int)] >= burst_threshold]

        min_fr_during_burst = brt.above_noise(burst_peaks, weighted_network_sdf)
        if min_fr_during_burst > 7:
            #min_fr_during_burst = min_fr_during_burst*0.1
            min_fr_during_burst = min_fr_during_burst * 0.35
        burst_peaks = burst_peaks[weighted_network_sdf[(burst_peaks * fs).astype(int)] > min_fr_during_burst]
        burst_borders = brt.detect_burst_borders(burst_peaks, weighted_network_sdf, threshold=min_fr_during_burst)

        # Is Network reverberating based on clustering?
        isReverb, percent_overlapping_IBI, percent_overlapping_FR, _, _, _, centers, dst = drc.k_means(burst_peaks, burst_borders,
                                                                                weighted_network_sdf)
        '''
        Detect if reverberating and calculate appropriate features
        '''
        if ((len(burst_peaks) > 3)):
            # If network IBI is not right skew and clustering suggests not reverberating
            if (brt.burst_skewness_reverberating(burst_peaks) == False) & (isReverb == False):
                rmax = 0.0001
                # sb_start, sb_end, num_reverbs = brt.detect_reverberation(prime_burst_peaks, weighted_network_sdf, rmax)
                num_reverbs = np.nan
                sb_start = [x[0] for x in burst_borders]
                sb_end = [x[1] for x in burst_borders]

                sb_duration = brt.super_burst_duration(sb_start, sb_end)
                isbi = brt.inter_super_burst_interval(sb_start, sb_end)
            else:
                try:
                    rmax = brt.find_rmax_hist(burst_peaks, prime_burst_peaks)
                    sb_start, sb_end, num_reverbs = brt.detect_reverberations_merge2(burst_borders, burst_peaks, prime_burst_peaks, rmax)
                    partial_burst_peaks = brt.find_partial_reverb(weighted_network_sdf, burst_threshold, 0.5)
                    iri = brt.inter_reverberation_interval(burst_peaks, sb_start, sb_end)
                    sb_duration = brt.super_burst_duration(sb_start, sb_end)
                    isbi = brt.inter_super_burst_interval(sb_start, sb_end)
                    print(f"Well had {np.count_nonzero(num_reverbs)} potential reverberating bursts.")
                    adaptation = brt.burst_adaptation(burst_peaks, sb_start, sb_end)
                except:
                    print("This potentially reverberating well could not be analyzed.")
                    rmax = 0.0001
                    num_reverbs = np.nan
                    sb_start = [x[0] for x in burst_borders]
                    sb_end = [x[1] for x in burst_borders]

                    sb_duration = brt.super_burst_duration(sb_start, sb_end)
                    isbi = brt.inter_super_burst_interval(sb_start, sb_end)
            fraction_of_participating_electrodes, burst_strength, burst_window = brt.reverb_strength(raster, burst_peaks, gaussian_kernel)
            synchrony = brt.compute_synchrony_reverb(raster, sb_start, sb_end)
        else:
            try:
                rmax = 0.0001
                num_reverbs = np.nan
                sb_start = [x[0] for x in burst_borders]
                sb_end = [x[1] for x in burst_borders]

                sb_duration = brt.super_burst_duration(sb_start, sb_end)
                isbi = brt.inter_super_burst_interval(sb_start, sb_end)
                fraction_of_participating_electrodes, burst_strength, burst_window = brt.reverb_strength(raster,
                                                                                                         burst_peaks,
                                                                                                         gaussian_kernel)
                synchrony = brt.compute_synchrony_reverb(raster, sb_start, sb_end)
            except:
                print(f"Skipping this well.")
    '''
    well_reverb = [isReverb, prime_burst_peaks, burst_peaks, partial_burst_peaks, burst_borders, rmax, sb_start, sb_end,
                   num_reverbs, iri, sb_duration, isbi, fraction_of_participating_electrodes, burst_strength,
                   burst_window, adaptation, synchrony, STTC_matrix]'''

    well_reverb = [isReverb, prime_burst_peaks, burst_peaks, partial_burst_peaks, burst_borders, rmax, sb_start, sb_end,
                   num_reverbs, iri, sb_duration, isbi, fraction_of_participating_electrodes, burst_strength,
                   burst_window, adaptation, synchrony, STTC_matrix, weighted_network_sdf, raster]
    return well_reverb
