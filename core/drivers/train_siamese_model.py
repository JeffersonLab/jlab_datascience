from datetime import datetime
# Set fixed seed values for numpy, os, random, and Tensorflow
seed_value = 0
# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os

os.environ['PYTHONHASHSEED'] = str(seed_value)
# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random

random.seed(seed_value)
# 3. Set the `numpy` pseudo-random generator at a fixed value
import numpy as np

np.random.seed(seed_value)
# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf

tf.random.set_seed(seed_value)
print("TF version: ", tf.__version__)

import core.keras_models.siamese_models as sms
from core.keras_models.callbacks import ResetCovarianceCallback
import core.keras_models.losses as losses
import core.keras_dataprep.siamese_generator as sg
import tensorflow.keras as keras

data_dir = '/work/datascience/'
train_generator = sg.SiameseDataGenerator(
    data_dir + 'errantbeam_train_ftype00000000_nsamples10000_npeaks75_indexaligned_mediancuts_dtypef32_sept2020_v3.npy',
    data_dir + 'errantbeam_train_ftype00110000_nsamples10000_npeaks75_indexaligned_mediancuts_dtypef32_sept2020_v3.npy',
    samples_per_anomaly_batch=128)
val_generator = sg.SiameseDataGenerator(
    data_dir + 'errantbeam_val_ftype00000000_nsamples10000_npeaks75_indexaligned_mediancuts_dtypef32_sept2020_v3.npy',
    data_dir + 'errantbeam_val_ftype00110000_nsamples10000_npeaks75_indexaligned_mediancuts_dtypef32_sept2020_v3.npy',
    samples_per_anomaly_batch=128)

(xl, xr), y = train_generator.__getitem__(1)
trace_length = xl.shape[1]
print('trace_length:', trace_length)
print("Number of traces per batch: ", len(xl))

distance_metric, drop_percentage, lambda_shift, alpha = "l2", 0.1, -5.0, 5.2

my_strategy = tf.distribute.MirroredStrategy()
with my_strategy.scope():
    model = sms.SiameseModelWithResNet(UQ=False, num_filters=[16, 32, 64, 128],
                                       strides=[2, 2, 2, 2], CommonDense_nodes=[128], CommonDrop_percentage=[0.095],
                                       distanceMetric=distance_metric,
                                       input_shape=(trace_length, 1), dropPercentage=drop_percentage,
                                       lambdaShift=lambda_shift, **{'activation': "sigmoid"})

    opt = keras.optimizers.Adam(learning_rate=1e-5)
    model.compile(loss=losses.closs(margin=1, alpha=2), optimizer=opt, metrics=['accuracy'])
    model.summary()

now = datetime.now()
timestamp = now.strftime("D%m%d%Y-T%H%M%S")
checkpoint_filepath = '/work/data_science/kishan/model-checkpoints/EB/sngp_siamese-tfdata-{}-v1'.format(timestamp)
model_ckpt = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_best_only=False,
    save_weights_only=False,
    save_freq="epoch",
    monitor='val_accuracy',
    mode='max')
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.85, patience=5, min_lr=1e-6, verbose=1)
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto',
                                               baseline=None, restore_best_weights=True)

fit_config = dict(epochs=400)

history = model.fit(train_generator,
                    validation_data=val_generator,
                    epochs=500,
                    callbacks=[model_ckpt, reduce_lr, early_stopping, ResetCovarianceCallback()])

model.save("/work/data_science/kishan/trainedModels/EB_SNGP_SeptData_gen_Model")
