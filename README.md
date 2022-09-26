# Burst-Reverberation-Toolbox
Toolbox for the detection and analysis of reverberating bursts in spiketrains


This is a beta version of Python package to detect and analyze reverberating bursts in electrophysiological recordings. The detection involves four steps:
1. Pre-processing
2. Spike density function
3. Burst detection
4. Burst reverberation detection
Once reverberating bursts are detected, analysis and generation of features are performed.

## 1. Pre-processing
Removes metadata from Axion MEA spikelist (mixed-bag of chronologically order spikes across 12 plates x 64 channels) and organizes a spike list of times for each channel for each well. 

## 2. Spike denisty function
Spike times  for each channel are converted to a binary spike matrix. Binary spike matrices are convolved with a guassian function (kernel width = 0.075s). <i>In future versions, kernel width is adaptively determined based on inter-spike-intervals (ISIs).</i> Convolution of kernel with spikematrix generates a spike density function, which is used as an estimate of instantaneous firing rate of the channel. Weighted mean firing rate for the network is calculated by taking the firing-rate-weighted average of all channels within a network.

## 3. Burst detection
Prime bursts are defined as periods where the network firing rate has a prominence greater than 40% of max firing rate (when max firing rate is greater than 5Hz) - identified as being robust network bursts. Potential bursts are identified as periods where the network firing rate is greater than noise with a prominence greater than deviations from the noise. <i>Improved signal-to-noise measure is implemented in future versions.</i> Potential bursts are combined with prime bursts to generate all bursts.

## 4. Burst reverberation detection
This approach involves three steps: first, inter-burst-interval and firing rate for each burst is calculated, second, a K-means clustering approach is used to identify clusters, and third, a procedural analysis is used to validate step 2 and generate features.

1. Burst reverberations are defined as periods where the firing rate of the network has repeated rise-and-falls (i.e., bursts followed by quiescence) following an initial spike in activity. The inter-burst-interval (IBI) within a reverberation is considerably faster than the IBI between "super-bursts" (network bursts that contain reverberations). Firing rate of the initial burst is always larger than the firing rate of the reverberating bursts. Based on these two observations, IBI and firing rate was calculated for all bursts.
2. K-means clustering was used to identify clusters in a 2-dimensional space (IBI, firing rate). Reverberating networks contained two clear, separable clusters while non-reverberating networks contained either one cluster or two clusters that overlapped in one of the dimensions.
3. Procedural analysis based on burst distribution bimodality and skewness was used to validate the previous approach.
