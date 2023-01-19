# Author: Kishansingh Rajput, Malachi Schram
# Org: Thomas Jefferson National Accelerator Facility
# Script: Training script for Errant Beam fault prediction models

# Set fixed seed values for numpy, os, random, and Tensorflow
# Apparently you may use different seed values at each stage
import random
import argparse
import sys

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os

# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random

# 3. Set the `numpy` pseudo-random generator at a fixed value
import numpy as np

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
print("TF version: ", tf.__version__)
# import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import core.keras_models.losses as losses
from core.keras_models.siamese_models_v2 import siameseModelWithResNet
import matplotlib.pyplot as plt
from core.keras_models.callbacks import ResetCovarianceCallback
# from core.visualization.ML_vis import plt_metric, getROC_Band, getSeparationPlot, getROC
from sklearn.metrics import roc_curve
import core.keras_dataprep.siamese_generator_v2 as sg


def run_training(decoder=None, output_dir="./", seed_value=0):
    
    # Set seeds
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "seed.txt"), "w") as f:
        f.write(str(seed_value))
    data_dir='/work/data_science/suf_sns/upstream_data/'
    train_generator = sg.SiameseDataGenerator(data_dir+'errantbeam_normal_train_traceid_Trace2_ns10.0_k_np96_nshift-1_Oct21AndFeb22_timestampFiltered_v0.npy',
                                              data_dir+'errantbeam_abnormal_train_traceid_Trace2_ns10.0_k_np96_nshift-1_Oct21AndFeb22_timestampFiltered_v0.npy',
                                              samples_per_anomaly_batch=15,
                                             num_anomalies_batch=10,
                                             min_max_norm=[0., 0.04],
                                             decoder=decoder)
    val_generator = sg.SiameseDataGenerator(data_dir+'errantbeam_normal_val_traceid_Trace2_ns10.0_k_np96_nshift-1_Oct21AndFeb22_timestampFiltered_v0.npy',
                                            data_dir+'errantbeam_abnormal_val_traceid_Trace2_ns10.0_k_np96_nshift-1_Oct21AndFeb22_timestampFiltered_v0.npy',
                                            samples_per_anomaly_batch=15,
                                           num_anomalies_batch=10,
                                            min_max_norm=[0., 0.04],
                                           decoder=decoder)
    test_generator = sg.SiameseDataGenerator(data_dir+'errantbeam_normal_test_traceid_Trace2_ns10.0_k_np96_nshift-1_Oct21AndFeb22_timestampFiltered_v0.npy',
                                            data_dir+'errantbeam_abnormal_test_traceid_Trace2_ns10.0_k_np96_nshift-1_Oct21AndFeb22_timestampFiltered_v0.npy',
                                            samples_per_anomaly_batch=15,
                                            num_anomalies_batch=10,
                                            min_max_norm=[0., 0.04],
                                            decoder=decoder)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = siameseModelWithResNet(UQ=True, decoder=decoder, num_filters=[16, 32, 64, 128],
                                       strides=[2,2,2,2], CommonDense_nodes=[128], CommonDrop_percentage=[0.05], distanceMetric="l2",
                                       input_shape=(10000, 1), dropPercentage=0.05, lambdaShift=-5, **{'activation':"sigmoid"})

        opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
        model.compile(loss=[losses.closs(margin=1, alpha=5.2), "mae"], optimizer=opt)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.85, patience=5, min_lr=1e-6,verbose=1)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0, patience=20, verbose=1, mode='auto',
                                   baseline=None, restore_best_weights=True)

    # Train model on dataset
    history = model.fit(train_generator,
                            validation_data=val_generator,
                            epochs=500,
                            callbacks=[reduce_lr, early_stopping, ResetCovarianceCallback()])

    model.save(os.path.join(output_dir, "SiameseWithoutAutoEncoder_2022.12.06_OctFeb_WithNormSeed"+str(seed_value)))
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoder", help="Which side to put the decoder or No decoder",
                        type=str, default='None')
    parser.add_argument("--output_directory", help="Output directory to save the results",
                        type=str, default='./')
    parser.add_argument("--seed", help="seed value for random operations",
                        type=int, default=0)


    # Get input arguments
    args = parser.parse_args()

    
    decoder = args.decoder
    output_directory = args.output_directory
    seed = args.seed
    
    if decoder.lower() not in ['left', 'right']:
        decoder = None
    else:
        decoder = decoder.lower()
    seed = int(seed)
    
    # Print input settings
    print('\nUsing decoder:', decoder)
    print('Using output directory:', output_directory)
    print('Using Seed value:', seed)
    
    run_training(decoder=decoder, output_dir=output_directory, seed_value=seed)
