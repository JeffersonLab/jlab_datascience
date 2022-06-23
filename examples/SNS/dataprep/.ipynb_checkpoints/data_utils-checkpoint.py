import tensorflow as tf

# Get individual trace
from sns_rad.binary import ReadStorage
import numpy as np


def test():
    print('import test worked')


def get_traces(filename, var_id='Trace2', begin=3600, shift=4000, data_type=0, bWidth_cut=900):
    file = ReadStorage(filename)
    end = begin + shift
    traces = []
    timestamps = []
    # Miha's cut
    for record in file:
        if 'bWidth' in record['parameters'].keys():
            bWidth_val  = record['parameters']['bWidth']
            if bWidth_val < bWidth_cut:
                print('bWidth below threshold:',bWidth_val)
                return np.array(traces)

    # print('get_traces: ',len(file))
    # if len(file) == 145:
    for idx in range(len(file)):
        if var_id in file[idx]['name']:
            trace = np.copy(file[idx]['value'][begin:end])
            if 'Before' == file[idx]['tags'][0] and data_type == -1:
                traces.append(trace)
                timestamps.append(file[idx]['timestamp'])
            if 'Before' in file[idx]['tags'][0] and data_type == 0:
                traces.append(trace)
                timestamps.append(file[idx]['timestamp'])
            if 'During' in file[idx]['tags'][0] and data_type == 1:
                traces.append(trace)
                timestamps.append(file[idx]['timestamp'])
    # else:
    #    print('Incorrect number of traces')
    np_traces = np.array(traces)
    return np_traces, timestamps


def load_dcm_traces(filename, scan_begin=0, scan_shift=25000, data_type=0, nfiles=-1):
    traces1, traces2 = [], []
    traces1 = get_traces(filename, var_id='Trace1', begin=scan_begin, shift=scan_shift, data_type=data_type)
    traces2 = get_traces(filename, var_id='Trace2', begin=scan_begin, shift=scan_shift, data_type=data_type)
    if traces1.shape[0] == 0:
        return [], []
    else:
        return traces1, traces2


def get_diff_matrix(one_trace, threshold=1e-9):
    import dask.array as da
    size = one_trace.shape[0]
    one_traceT = one_trace.reshape(-1, 1)
    one_trace = one_trace.reshape(1, -1)
    da_one_trace = da.array(one_trace)
    da_one_traceT = da.array(one_traceT)
    maxtrix_ones = da.ones((size, size))
    da_one_trace_m = maxtrix_ones * da_one_trace
    da_one_trace_mT = maxtrix_ones * da_one_traceT
    matrix_diff = da.log(abs(da_one_trace_m - da_one_trace_mT) + threshold)
    return matrix_diff


def create_diff_dataset(dataset, look_back=1, look_forward=1):
    """
    Prepare dataset for the time series RNN.
    Params
    ------
    dataset: np.array
        Time series data
    look_back: int
        History for the LSTM model
    look_forward: int
        Number of future instances
    Return
    -------
    (np.array, np.array): Feature matrix and output array
    """
    dataset_shape = dataset.shape
    x, y = [], []
    offset = look_back + look_forward
    for i in range(len(dataset) - (offset + 1)):
        xx = dataset[i:(i + look_back), i:(i + look_back)]
        yy = dataset[(i + look_back):(i + offset), (i + look_back):(i + offset)]
        x.append(xx)
        y.append(yy)
    return np.array(x), np.array(y)


def create_dataset(dataset, look_back=1, look_forward=1):
    """
    Prepare dataset for the time series RNN.
    Params
    ------
    dataset: np.array
        Time series data
    look_back: int
        History for the LSTM model
    look_forward: int
        Number of future instances
    Return
    -------
    (np.array, np.array): Feature matrix and output array
    """
    x, y = [], []
    offset = look_back + look_forward
    for i in range(len(dataset) - (offset + 1)):
        xx = dataset[i:(i + look_back), 0]
        yy = dataset[(i + look_back):(i + offset), 0]
        x.append(xx)
        y.append(yy)
    return np.array(x), np.array(y)


def prep_data(trace, look_back, look_forward):
    X_train, Y_train = create_dataset(trace, look_back, look_forward)
    X_train = np.expand_dims(X_train, axis=2)
    Y_train = np.expand_dims(Y_train, axis=2)
    return X_train, Y_train


def concate_traces(datasets, look_back=1, look_forward=1):
    """
    Prepare datasets for the time series RNN from multiple datasets.
    Params
    ------
    datasets: np.array
        Collection of time series data for a fixed set of variables
    look_back: int
        History for the RNN model
    look_forward: int
        Number of future instances
    Return
    -------
    (np.array, np.array): Feature matrix and output array
    """
    x_train_list, y_train_list = [], []
    for trace in datasets:
        x_train, x_train = create_dataset(trace.reshape(-1, 1), look_back, look_forward)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1]))
        x_train_list.append(x_train)
        y_train_list.append(y_train)
    full_x_train = np.concatenate(x_train_list, axis=0)
    full_y_train = np.concatenate(y_train_list, axis=0)
    return full_x_train, full_y_train


class DatasetGeneratorFS(tf.data.Dataset):
    def _generator(dataset, look_back, look_forward):
        # creating window data
        offset = look_back + look_forward
        for i in range(len(dataset) - (offset + 1)):
            xx = dataset[i:(i + look_back), 0]
            yy = dataset[(i + look_back):(i + offset), 0]

            # yield the row
            yield (np.array(xx).reshape(-1, 1), np.array(xx).reshape(-1, 1)), (
                np.array(xx).reshape(-1, 1), np.array(yy).reshape(-1, 1))

    def __new__(self, dataset, look_back, look_forward):
        return tf.data.Dataset.from_generator(
            self._generator,
            output_types=((tf.float32, tf.float32), (tf.float32, tf.float32)),
            output_shapes=(((look_back, 1), (look_back, 1)), ((look_back, 1), (look_forward, 1))),
            args=(dataset, look_back, look_forward,)
        )


class DatasetGeneratorAE(tf.data.Dataset):
    def _generator(dataset, look_back):
        # creating window data
        offset = look_back
        for i in range(len(dataset) - (offset + 1)):
            xx = dataset[i:(i + look_back), 0]
            # yield the row
            yield (np.array(xx).reshape(-1, 1), np.array(xx).reshape(-1, 1))

    def __new__(self, dataset, look_back):
        return tf.data.Dataset.from_generator(
            self._generator,
            output_types=((tf.float32, tf.float32)),
            output_shapes=((look_back, 1), (look_back, 1)),
            args=(dataset, look_back,)
        )


class DatasetGeneratorAEFS(tf.data.Dataset):
    """
     Data generator for auto-encoder + few shot classification
     Params
     ------
     """

    def _generator(dataset, look_back, look_forward):
        # creating window data
        offset = look_back + look_forward
        for i in range(len(dataset) - (offset + 1)):
            xx = dataset[i:(i + look_back), 0]
            yy = dataset[(i + look_back):(i + offset), 0]

            # yield the row
            yield (np.array(xx).reshape(-1, 1), np.array(xx).reshape(-1, 1)), (
                np.array(xx).reshape(-1, 1), np.array(yy).reshape(-1, 1))
            yield (np.array(xx).reshape(-1, 1), np.array(xx).reshape(-1, 1))

    def __new__(self, dataset, look_back, look_forward):
        return tf.data.Dataset.from_generator(
            self._generator,
            output_types=(tf.float32, tf.float32, tf.float32),
            output_shapes=((look_back, 1), (look_back, 1), (1,)),
            args=(dataset, look_back, look_forward,)
        )