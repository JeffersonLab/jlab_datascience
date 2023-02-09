import mlflow
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

import core.keras_models.losses as losses
from core.keras_models.siamese_models_v2 import siameseModelWithResNet
from core.keras_models.callbacks import ResetCovarianceCallback
import core.keras_dataprep.siamese_generator_v2 as sg
# from suf_sns.visualization.ML_vis import plt_metric, getROC_Band, getSeparationPlot, getROC
from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random
from functools import partial
import shutil
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'

experiment_name = "Siamese_HPO_NAS"
try:
    experiment_id = mlflow.create_experiment(f"{experiment_name}")
    experiment = mlflow.get_experiment_by_name(f"{experiment_name}")
    mlflow.set_experiment(f"{experiment_name}")
except:
    try:
        experiment = mlflow.get_experiment_by_name(f"{experiment_name}")
        experiment_id = experiment.experiment_id
        mlflow.set_experiment(f"{experiment_name}")
    except Exception as err:
        print(f'error setting experiment: ' + err)

print("Name: {}".format(experiment.name))
print("Experiment_id: {}".format(experiment.experiment_id))
print("Artifact Location: {}".format(experiment.artifact_location))
print("Tags: {}".format(experiment.tags))
print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))

def get_predictions_deterministic_v0(model, generator, n):
    predictions, labels = [], []
    for i in range(n):
        (xl, xr), y = generator.__getitem__(i)
        preds = model((xl, xr))
        preds = np.array(preds)
        preds0, preds1 = np.mean(np.squeeze(preds[np.where(y==0)[0]])), np.mean(np.squeeze(preds[np.where(y==1)[0]]))
        predictions.extend([preds0, preds1])
        labels.extend([0., 1.])
    predictions, labels = np.array(predictions), np.array(labels)
    return predictions, labels

def get_predictions_deterministic(model, generator, n):
    predictions, labels = [], []
    for j in range(n):
        (xl, xr), y = generator.__getitem__(j)
        unique_normal = np.unique(xr[np.where(y==0)[0]], axis=0)
        unique_anomaly = np.unique(xr[np.where(y==1)[0]], axis=0)
        preds = model((xl, xr))
        preds = np.squeeze(np.array(preds))
        for i, normal in enumerate(unique_normal):
            anomaly = unique_anomaly[i]
            pred0 = np.mean(preds[np.where(xr == normal)[0]])
            pred1 = np.mean(preds[np.where(xr == anomaly)[0]])
            predictions.extend([pred0, pred1])
            labels.extend([0., 1.])
    predictions = np.array(predictions)
    labels = np.array(labels)
    return predictions, labels

def evaluate_model(predictions, labels):
    loss = np.mean(np.square(predictions-labels))
    accuracy = accuracy_score(labels, np.round(predictions))
    tpr = get_tpr(labels, predictions, 0.005)
    return loss, accuracy, tpr

def get_tpr(labels, predictions, false_positive_rate):
    fpr, tpr, ths = roc_curve(labels, predictions)
    return np.interp([0.005], fpr, tpr)[0]

def plt_metric(history, metric, title, has_valid=True, saveloc=None):
    """Plots the given 'metric' from 'history'.

    Arguments:
        history: history attribute of History object returned from Model.fit.
        metric: Metric to plot, a string value present as key in 'history'.
        title: A string to be used as title of plot.
        has_valid: Boolean, true if valid data was passed to Model.fit else false.

    Returns:
        None.
    """
    plt.plot(history[metric])
    if has_valid:
        plt.plot(history["val_" + metric])
        plt.legend(["train", "validation"], loc="upper left")
    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel("epoch")
    plt.yscale("log")
    if saveloc is not None:
        plt.savefig(saveloc)
    plt.show()
    
def getROC(labels, predictions, names, xlimit=[0, 1], ylimit=[0, 1], saveloc=None):
    plt.clf()
    fig= plt.figure(figsize=(8,6),dpi=100)
    from sklearn.metrics import roc_curve, roc_auc_score
    for i in range(len(labels)):
        lr_fpr_train, lr_tpr_train, _ = roc_curve(labels[i], predictions[i])
        auc_train = roc_auc_score(labels[i], predictions[i])
        plt.plot(lr_fpr_train, lr_tpr_train, linewidth=1.5, label=names[i]+" AUC:"+str(np.round(auc_train, 4)))
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(color='gray', linestyle='--', linewidth=1)
    plt.legend()
    ident = [0.0, 1.0]
    plt.plot(ident,ident, ls='--')
    plt.xlim(xlimit)
    plt.ylim(ylimit)
    if saveloc is not None:
        plt.savefig(saveloc)
    plt.show()

def _train_SNGP(params, hyperopt=False):
    # Set seeds
    seed_value = 0
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    tf.keras.utils.set_random_seed(seed_value)

    print("Evaluating on: ", params)
    
    # ExtractHPO Params
    alpha = params['alpha'] 
    bs = params['batch_size'] 
    num_anomalies, num_samples_per_anomaly = bs
    distance_metric = params['distance_metric']
    lambda_shift = params['lambda_shift']
    learning_rate = params['learning_rate']
    
    # ExtractNAS Params
    num_conv_blocks = params["num_conv_blocks"]
    num_filters = [params["num_filters"] for i in range(num_conv_blocks)]
    strides = [params["strides"] for i in range(num_conv_blocks)]
    kernel_size = [params["kernel_size"] for i in range(num_conv_blocks)]
    activation = [params["activation"] for i in range(num_conv_blocks)]
    dropouts = [params["dropouts"] for i in range(num_conv_blocks)]
    print(strides)
    
    num_dense_layers = params["num_dense_layers"]
    num_nodes_dense = [params["num_nodes_dense"] for i in range(num_dense_layers)]
    dense_activation = [params["dense_activation"] for i in range(num_dense_layers)]
    dense_dropouts = [params["dense_dropouts"] for i in range(num_dense_layers)]
    distance_dropout = params["distance_dropout"]
    
    
    
    
    print(alpha, bs, distance_metric, lambda_shift, learning_rate)
    num_anomalies, num_samples_per_anomaly = bs

    data_dir='/work/data_science/suf_sns/upstream_data/'
    train_generator = sg.SiameseDataGenerator(data_dir+'errantbeam_normal_train_traceid_Trace2_ns10.0_k_np96_nshift-1_Oct21AndFeb22_timestampFiltered_v0.npy',
                                              data_dir+'errantbeam_abnormal_train_traceid_Trace2_ns10.0_k_np96_nshift-1_Oct21AndFeb22_timestampFiltered_v0.npy',
                                              samples_per_anomaly_batch=num_samples_per_anomaly,
                                             num_anomalies_batch=num_anomalies,
                                             min_max_norm=[0., 0.04],
                                             decoder=None, seed=seed_value)
    val_generator = sg.SiameseDataGenerator(data_dir+'errantbeam_normal_val_traceid_Trace2_ns10.0_k_np96_nshift-1_Oct21AndFeb22_timestampFiltered_v0.npy',
                                            data_dir+'errantbeam_abnormal_val_traceid_Trace2_ns10.0_k_np96_nshift-1_Oct21AndFeb22_timestampFiltered_v0.npy',
                                            samples_per_anomaly_batch=num_samples_per_anomaly,
                                           num_anomalies_batch=num_anomalies,
                                            min_max_norm=[0., 0.04],
                                           decoder=None, seed=seed_value)
    test_generator = sg.SiameseDataGenerator(data_dir+'errantbeam_normal_test_traceid_Trace2_ns10.0_k_np96_nshift-1_Oct21AndFeb22_timestampFiltered_v0.npy',
                                            data_dir+'errantbeam_abnormal_test_traceid_Trace2_ns10.0_k_np96_nshift-1_Oct21AndFeb22_timestampFiltered_v0.npy',
                                            samples_per_anomaly_batch=num_samples_per_anomaly,
                                            num_anomalies_batch=num_anomalies,
                                            min_max_norm=[0., 0.04],
                                            decoder=None, seed=seed_value)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = siameseModelWithResNet(UQ=False, 
                                       decoder=None, 
                                       num_filters=num_filters,
                                       strides=strides,
                                       kernel_sizes=kernel_size,
                                       activations=activation,
                                       dropouts=dropouts,
                                       CommonDense_nodes=num_nodes_dense, 
                                       CommonDrop_percentage=dense_dropouts, 
                                       dense_activations=dense_activation,
                                       distance_dropout=distance_dropout,
                                       distanceMetric=distance_metric,
                                       input_shape=(10000, 1), 
                                       dropPercentage=0.05, 
                                       lambdaShift=lambda_shift, 
                                       **{'activation':"sigmoid"})

        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(loss=[losses.closs(margin=1, alpha=alpha), "mae"], metrics=['accuracy'], optimizer=opt)
        model.build((2, None, 10000, 1))
        model.summary()

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.85, patience=5, min_lr=1e-6,verbose=1)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0, patience=20, verbose=1, mode='auto',
                                   baseline=None, restore_best_weights=True)
    # Train model on dataset
    history = model.fit(train_generator,
                            validation_data=val_generator,
                            epochs=2,
                            callbacks=[reduce_lr, early_stopping])

    # model.save(os.path.join(output_dir, "SiameseWithoutAutoEncoder_2022.12.06_OctFeb_WithNormSeed"+str(seed_value)))
    plt_metric(history.history, metric='loss', title='Loss', saveloc="loss_curve.png")
    mlflow.log_artifact("loss_curve.png")
    plt.clf()

    train_generator = sg.SiameseDataGenerator(data_dir+'errantbeam_normal_train_traceid_Trace2_ns10.0_k_np96_nshift-1_Oct21AndFeb22_timestampFiltered_v0.npy',
                                              data_dir+'errantbeam_abnormal_train_traceid_Trace2_ns10.0_k_np96_nshift-1_Oct21AndFeb22_timestampFiltered_v0.npy',
                                              samples_per_anomaly_batch=15,
                                             num_anomalies_batch=1,
                                             min_max_norm=[0., 0.04],
                                             decoder=None, seed=seed_value)
    val_generator = sg.SiameseDataGenerator(data_dir+'errantbeam_normal_val_traceid_Trace2_ns10.0_k_np96_nshift-1_Oct21AndFeb22_timestampFiltered_v0.npy',
                                            data_dir+'errantbeam_abnormal_val_traceid_Trace2_ns10.0_k_np96_nshift-1_Oct21AndFeb22_timestampFiltered_v0.npy',
                                            samples_per_anomaly_batch=15,
                                           num_anomalies_batch=1,
                                            min_max_norm=[0., 0.04],
                                           decoder=None, seed=seed_value)
    test_generator = sg.SiameseDataGenerator(data_dir+'errantbeam_normal_test_traceid_Trace2_ns10.0_k_np96_nshift-1_Oct21AndFeb22_timestampFiltered_v0.npy',
                                            data_dir+'errantbeam_abnormal_test_traceid_Trace2_ns10.0_k_np96_nshift-1_Oct21AndFeb22_timestampFiltered_v0.npy',
                                            samples_per_anomaly_batch=15,
                                            num_anomalies_batch=1,
                                            min_max_norm=[0., 0.04],
                                            decoder=None, seed=seed_value)


    # Test the models
    train_preds, train_labels = get_predictions_deterministic(model, train_generator, int(5120))
    val_preds, val_labels = get_predictions_deterministic(model, val_generator, int(1280))
    test_preds, test_labels = get_predictions_deterministic(model, test_generator, int(1600))
    train_loss, train_acc, train_tpr = evaluate_model(train_preds, train_labels)
    val_loss, val_acc, val_tpr = evaluate_model(val_preds, val_labels)
    test_loss, test_acc, test_tpr = evaluate_model(test_preds, test_labels)

    getROC([val_labels, test_labels],
           [val_preds, test_preds],  
           ['validation set', 'test set'],
            xlimit=[0., 1.],
            saveloc="ROC_full.png")
    mlflow.log_artifact("ROC_full.png")
    plt.clf()

    getROC([val_labels, test_labels],
           [val_preds, test_preds],  
           ['validation set', 'test set'],
            xlimit=[0., 0.005],
            saveloc="ROC_zoomed.png")
    mlflow.log_artifact("ROC_zoomed.png")
    plt.clf()

    
    mlparams = {
        "distance_metric":str(distance_metric),
        "lambda_shift":str(lambda_shift),
        "alpha":str(alpha),
        "learning_rate":str(learning_rate),
        "batch_size":str(bs)
    }
    mlflow.log_params(mlparams)

    mlmetrics = {"training_loss": train_loss,
                "val_loss": val_loss,
                "test_loss":test_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "test_acc": test_acc,
                "train_tpr": train_tpr,
                "val_tpr": val_tpr,
                "test_tpr": test_tpr,
                }
    mlflow.log_metrics(mlmetrics)

    if (not hyperopt):
        return model
    return {'loss': -1*test_tpr, 'status': STATUS_OK}

# public train
def train_SNGP(params, hyperopt=False):
    """
    Proxy function used to call _train
    :param params: hyperparameters. Its structure is consistent with how search space is defined. See below.
    :param fpath: Path or URL for the training data used with the model.
    :param hyperopt: Use hyperopt for hyperparameter search during training.
    :return: dict with fields 'loss' (scalar loss) and 'status' (success/failure status of run)
    """
    with mlflow.start_run(nested=True):
        return _train_SNGP(params, hyperopt)


algorithm = 'tpe' # Best Hyperparameter Algorithm: Tree-structured Parzen Estimator
# PATH_FINAL_MODEL = os.path.join(os.path.dirname(os.path.abspath("__file__")),'final_model')


search_space = {
    #HPO Parameters
    "alpha":hp.uniform("alpha", 0.1, 10.),
    "batch_size":hp.choice("batch_size", [(5,10), (10,10), (15,10), (5, 20), (10, 20), (15,20), (5,40), (10,20), (15,40)]),
    "distance_metric":hp.choice("distance_metric", ['l1', 'l2']),
    "lambda_shift":hp.choice("lambda_shift", [-10., -5., 0., 5.]),
    "learning_rate":hp.choice("learning_rate", [1e-3, 1e-4, 1e-5, 1e-6]),
    
    
    #NAS Parameters
    "num_conv_blocks":hp.choice("num_conv_blocks", [2,3,4,5,6]),
    "num_filters":hp.choice("num_filters", [8, 16, 32, 64, 128, 256]),
    "strides":hp.choice("strides", [1,2]),
    "kernel_size":hp.choice("kernel_size", [2,3,4,5,6]),
    "activation":hp.choice("activation", ['relu', 'tanh', 'leaky_relu']),
    "dropouts":hp.uniform("dropouts", 0.0, 0.15),
    "num_dense_layers":hp.choice("num_dense_layers", [1,2,3,4]),
    "num_nodes_dense":hp.choice("num_nodes_dense", [32,64,128,256,512]),
    "dense_activation":hp.choice("dense_activation", ['relu', 'tanh', 'leaky_relu', "linear"]),
    "dense_dropouts":hp.uniform("dense_dropouts", 0.0, 0.15),
    "distance_dropout":hp.uniform("distance_dropout", 0.0, 0.15)
}


trials = Trials()
# you can randomly search for best hyperparameters: hyperopt.rand.suggest or use hyperopt.tpe.suggest for TPE. Here we use TPE.
algorithm = tpe.suggest if algorithm == 'tpe' else None
fn = partial(train_SNGP, hyperopt=True)
experid = 0

with mlflow.start_run():
    mlflow.set_tag("mlflow.runName", experiment_name)
    argmin = fmin(fn=fn,
                  space=search_space,
                  algo=algorithm,
                  max_evals=10,
                  trials=trials)

    print("==========================================")
    fn = partial(train_SNGP, hyperopt=False)
    print(argmin)
    args = tuple(argmin.values())
    
    final_model = fn(args)
    FINAL_MODEL_DIR = 'final_model'
    # over-write any previous final model
    # need to delete directory if it already exists
    try:
        shutil.rmtree(FINAL_MODEL_DIR)
    except:
        pass
    os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
    
    # save the final model
    final_model.save(FINAL_MODEL_DIR)



