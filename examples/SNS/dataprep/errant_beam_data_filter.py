import os
import argparse
import random
import sys

import numpy as np
from sklearn.model_selection import train_test_split
import examples.SNS.dataprep.data_utils as data_utils
from scipy.signal import find_peaks
from sklearn.utils import shuffle

random.seed(30)
np.random.seed(30)

'''
Example:
python SNS/dataprep/errand_beam_filter.py --root_directory /Users/schram/globus_data/DCM/Errant/ --num_anomalies 4000 --samples_per_anomaly 30 --nsamples 10000 --npeaks 100 
'''
def filter_files(root_director, var_id='Trace1', num_anomalies=100, samples_per_anomaly=10, shift=10000, npeaks_cut=100, nshift=-1):
    rootdir = root_director
    filtered_files = []
    train_test_fraction = 0.2
    train_val_fraction = 0.2
    tFraction = train_val_fraction*(1-train_test_fraction)

    for root, subfolders, files in os.walk(rootdir):
        filtered_files += [os.path.join(root, file) for file in files if ('.bin' in file and ('DCML' not in file and '202202' in file))]

    print('Number of available files: ', len(filtered_files))

    filtered_normal_traces = []
    filtered_anomaly_traces = []
    filtered_normal_timestamps = []
    filtered_anomaly_timestamp = []
    filtered_files_wtraces = []
    for filename in filtered_files:
        print(filename)
        try:
            traces1, timestamps = data_utils.get_traces(filename, var_id=var_id, begin=3000, shift=shift, data_type=0)
        except Exception as e:
            traces1 = []
            print("Error in reading the file: ", filename)
            print("Error:", e)
        if len(traces1) >= abs(nshift):
            peaks, _ = find_peaks(traces1[nshift].flatten(), distance=75, height=0.002)
            npeaks = len(peaks)
            median = np.median(traces1[nshift])
            q25 = np.quantile(traces1[nshift],0.25)
            timestamp = timestamps[nshift]
            if npeaks < npeaks_cut or median < 0.02 or q25 > 0.0075:
                print('Problem with file - npeaks/median:', npeaks,'/', median)
                continue
            filtered_files_wtraces.append(filename)
            if '00000000' in filename and (len(filtered_normal_traces) < 2*num_anomalies or len(filtered_normal_traces) < int(float(samples_per_anomaly)/tFraction)+1):
                for trace in traces1:
                    npeaks = len(find_peaks(trace.flatten(), distance=75, height=0.002)[0])
                    if npeaks>=npeaks_cut:
                        filtered_normal_traces += [trace]
                        filtered_normal_timestamps += [timestamp]
                #filtered_normal_traces += [traces1[nshift]]#[trace for trace in traces1 if len(find_peaks(trace[0].flatten(), distance=75, height=0.002)[0]) < npeaks_cut]
                print('#normal:', len(filtered_normal_traces))
            elif '00110000' in filename and len(filtered_anomaly_traces) < num_anomalies:
                filtered_anomaly_traces += [traces1[nshift]]
                filtered_anomaly_timestamp += [timestamp]
            else:
                print('Omitting for now')
                if len(filtered_normal_traces) >= num_anomalies and len(filtered_anomaly_traces) >= num_anomalies and len(filtered_normal_traces) >= int(float(samples_per_anomaly)/tFraction+1):
                    break
        else:
            print('Invalid file')

    nfiltered_nomaly_traces = len(filtered_normal_traces)
    print('Number of normal total traces before timestamp filtering: ', nfiltered_nomaly_traces)

    normal_traces_contaminated = []
    for i in range(len(filtered_normal_timestamps)):
        for j in range(len(filtered_anomaly_timestamp)):
            diff = filtered_anomaly_timestamp[j] - filtered_normal_timestamps[i]
            if diff.total_seconds() <= 1 and diff.total_seconds() >= 0:
                normal_traces_contaminated += [i]

    normal_traces_contaminated = sorted(normal_traces_contaminated, reverse=True)
    for idx in normal_traces_contaminated:
        if idx < len(filtered_normal_traces):
            filtered_normal_traces.pop(idx)

    nfiltered_nomaly_traces = len(filtered_normal_traces)
    print('Number of normal total traces after timestamp filtering: ', nfiltered_nomaly_traces)

    nfiltered_anomaly_traces = len(filtered_anomaly_traces)
    print('Number of anomaly total traces: ', nfiltered_anomaly_traces)

    if num_anomalies > nfiltered_anomaly_traces:
        num_anomalies = nfiltered_anomaly_traces
        print('Number of anomaly traces used:', num_anomalies)
        filtered_normal_traces = filtered_normal_traces[:num_anomalies]

    filtered_normal_traces = filtered_normal_traces[:num_anomalies]
    nfiltered_normal_traces = len(filtered_normal_traces)
    print('Using number of normal total traces: ', nfiltered_normal_traces)
    #return -11

    if samples_per_anomaly > nfiltered_normal_traces:
        print("Requested number of samples per anomaly are: ", samples_per_anomaly)
        print("Total number of normal traces found are: ", nfiltered_normal_traces) # ", So updating samples per anomaly from to ", nfiltered_normal_traces)
        print("Exiting...")
        return

    train_normal_traces, test_normal_traces = train_test_split(filtered_normal_traces, test_size=train_test_fraction)
    train_abnormal_traces, test_abnormal_traces = train_test_split(filtered_anomaly_traces, test_size=train_test_fraction)

    train_normal_traces, val_normal_traces = train_test_split(train_normal_traces, test_size=train_val_fraction)
    train_abnormal_traces, val_abnormal_traces = train_test_split(train_abnormal_traces, test_size=train_val_fraction)

    print("Total Samples Found: ")
    print("\t\tGood\t\tBad")
    print("\t\t", nfiltered_normal_traces, "\t\t", nfiltered_anomaly_traces)
    print("Splitting")
    print("Train:")
    print("\t\t", len(train_normal_traces), "\t\t", len(train_abnormal_traces))
    print("Validation:")
    print("\t\t", len(val_normal_traces), "\t\t", len(val_abnormal_traces))
    print("Test:")
    print("\t\t", len(test_normal_traces), "\t\t", len(test_abnormal_traces))

    # Save before combinatoric
    pre_postfix = 'traceid_{}_ns{}_k_np{}_nshift{}_Feb22_timestampFiltered_Trace2_nonDCML_v0'.format(var_id, shift/1000.0, npeaks_cut, nshift)
    np.save('errantbeam_normal_train_' + pre_postfix + '.npy', np.array(train_normal_traces, np.float32))
    np.save('errantbeam_normal_val_' + pre_postfix + '.npy', np.array(val_normal_traces, np.float32))
    np.save('errantbeam_normal_test_' + pre_postfix + '.npy', np.array(test_normal_traces, np.float32))

    np.save('errantbeam_abnormal_train_' + pre_postfix + '.npy', np.array(train_abnormal_traces, np.float32))
    np.save('errantbeam_abnormal_val_' + pre_postfix + '.npy', np.array(val_abnormal_traces, np.float32))
    np.save('errantbeam_abnormal_test_' + pre_postfix + '.npy', np.array(test_abnormal_traces, np.float32))
    #sys.exit(-99)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_directory", help="Root directory used to scan the files",
                        type=str, default='/Users/kishan/ML4SNS/2021_DATA/')
    parser.add_argument("--num_anomalies", help="Number of anomalies included in sample",
                        type=int, default=3)
    parser.add_argument("--samples_per_anomaly", help="Number of normal samples used to compare a single anomaly",
                        type=int, default=2)
    parser.add_argument("--nsamples", help="Number of samples in trace",
                        type=int, default=10000)
    parser.add_argument("--nshifts", help="Number of shifted trace",
                        type=int, default=-1)
    parser.add_argument("--npeaks_cut", help="Number of peaks in trace",
                        type=int, default=95)
    parser.add_argument("--trace_name", help="The trace name to access",
                        type=str, default='Trace2')

    # Get input arguments
    args = parser.parse_args()
    root_directory = args.root_directory
    nsamples = args.nsamples
    npeaks_cut = args.npeaks_cut
    num_anomalies = args.num_anomalies
    samples_per_anomaly = args.samples_per_anomaly
    nshifts = args.nshifts
    trace_name = args.trace_name

    # Print input settings
    print('\nUsing root directory:', root_directory)
    print('Using trace name:', trace_name)
    print('Using number of samples in trace:', nsamples)
    print('Using number of peaks in trace:', npeaks_cut)
    print('Using number of anomalies:', num_anomalies)
    print('Using samples_per_anomaly:', samples_per_anomaly)
    print('Using shifted trace:', nshifts)
    filter_files(root_directory, trace_name, num_anomalies, samples_per_anomaly, nsamples, npeaks_cut, nshifts)
