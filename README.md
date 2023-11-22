# RESIST_perf_clustering
trying hierarchical clustering over performance matrix

This repo provides a first script to create a performance matrix of different variants of x264 executed over different videos.
Then, we apply hierarchical clustering and try to analyse generated clusters.

## Set-up
The scripts are written in Python3 (v3.8).
They use different libraries: numpy (written with v1.18) and pandas (written with v1.3) to manage structures and I/O operations over csv files.
They also use scipy (written with v1.7) and matplotlib (written with v3.4) to create and managing graphical plots (ie, dendograms and display of dendograms).
The cluster per se is done via scikit-learn (v0.24)
Please be sure that these librairies are installed.

## Folders
You will find two folders in this repo:
_data_ that contains all measures observed from different configurations of x264 when executed on different inputs.
The folder is structured per datasets. The subfolder _res\_ugc_ contains multiple files (one per video) in a csv format.
Inside each file (corresponding to a video), each line first describes the variants that have been used (one per line) and the last columns report the performance measures observed for different aspects.
8 different aspects have been observed from the size of the output video to the bitrate including the processing time.

_scripts_ gathers the different scripts that were developed to apply clustering on a dataset from the previous folder.
The scripts rely on the structure of the csv files and try to be as customizable as possible to try different parameter combinations from the clustering technique.

_results_ can be used to store dendograms from the different clustering configurations that were tried.

## Preliminary Results

### Configuration Recommendation from Nearest Neighbours in Input Feature Space

We look-up the k-nearest neighbour to an input and recommend that configuration that has the best average rank for these k inputs.
We then check how well this configuration performs for the input in terms of which rank it has (out of 201) and the MAPE (mean absolute percentage error).
The performance measure considered is the `elapsedtime`.
Nearest neighbour search is performed via the input features.
Validation is performed on 15% (194/1287 inputs) hold-out data (no cross-validation or control for randomness yet).

```
Avg. rank of 1 best recommended configuration: 8.67
Avg. MAPE of 1 best recommended configuration: 0.08
Avg. rank of 3 best recommended configuration: 3.39
Avg. MAPE of 3 best recommended configuration: 0.03
Avg. rank of 5 best recommended configuration: 2.46
Avg. MAPE of 5 best recommended configuration: 0.02
Avg. rank of 10 best recommended configuration: 1.36
Avg. MAPE of 10 best recommended configuration: 0.01
```
