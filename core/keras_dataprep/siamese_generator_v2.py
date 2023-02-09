import numpy as np
import tensorflow.keras as keras
import os
import pickle

class SiameseDataGenerator(keras.utils.Sequence):
    ' Generates siamese data for Keras '

    def __init__(self, good_file, bad_file, num_anomalies_batch=1, samples_per_anomaly_batch=1, shuffle=True, n_channels=1, before_index=-1, mpsLabels=None, min_max_norm=None, decoder=None, seed=0):
        'Initialization'
        os.environ['PYTHONHASHSEED']=str(seed)
        np.random.seed(seed)

        self.shuffle = shuffle
        self.samples_per_batch = samples_per_anomaly_batch
        self.num_anomalies_batch = num_anomalies_batch
        self.batch_size = samples_per_anomaly_batch * num_anomalies_batch

        self.normal_traces = np.load(good_file)
        self.normal_size = self.normal_traces.shape[0]
        self.normal_samples = self.normal_traces.shape[1]
        self.mpsLabels = mpsLabels
        self.decoder = decoder
        
        if mpsLabels != None:
            with open(mpsLabels, "rb") as f:
                fault_names = pickle.load(f)
            fault_name_id = {}
            id_num = 0
            for fault in fault_names:
                if fault not in fault_name_id:
                    fault_name_id[fault] = id_num
                    id_num += 1
            self.fault_ids = []
            for fault in fault_names:
                self.fault_ids.append(fault_name_id[fault])
            self.fault_ids = np.array(self.fault_ids)
 

        normal_per_anomaly_batch = int(np.floor(self.normal_size/self.num_anomalies_batch))-1
        if self.samples_per_batch > normal_per_anomaly_batch:
            self.samples_per_batch = normal_per_anomaly_batch
            print('Reducing samples per anomaly per batch to:',self.samples_per_batch)
        
        self.anomaly_traces = np.load(bad_file)
        if len(self.anomaly_traces.shape) != 2:
            self.anomaly_traces = self.anomaly_traces[:, before_index, :]
        self.anomaly_size = self.anomaly_traces.shape[0]
        
        if min_max_norm != None:
            mn, mx = min_max_norm[0], min_max_norm[1]
            self.normal_traces = (self.normal_traces - mn) / (mx-mn)
            self.anomaly_traces = (self.anomaly_traces - mn) / (mx-mn)
            
        self.anomaly_indices = np.arange(len(self.anomaly_traces))
        self.normal_indices = np.arange(len(self.normal_traces))
        
        print("Lengths: Normal (", len(self.normal_traces), ") Anomalies (", len(self.anomaly_traces), ")") #" fault ids (", len(self.fault_ids), ")")
        
        self.n_channels = n_channels
        self.normal_index = 0
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        self.num_batches = int(self.anomaly_size/self.num_anomalies_batch)
        return self.num_batches

    def __getitem__(self, index):
        'Generate one batch of data'
        
        # Generate indexes of the batch
        if index > self.anomaly_traces.shape[0]:
            print("Invalid index, returning")
            return
        
        indexes = self.anomaly_indices[index * self.num_anomalies_batch: (index + 1) * self.num_anomalies_batch]
        bad_traces = self.anomaly_traces[indexes]

        good_indexes = self.normal_indices[self.normal_index*self.num_anomalies_batch:(self.normal_index+1)*self.num_anomalies_batch]
        good_traces = self.normal_traces[good_indexes]
        
#         print('shapes: ', bad_traces.shape, good_traces.shape)
        self.normal_index += 1
        if (self.normal_index+1)*self.num_anomalies_batch > len(self.normal_indices):
            self.normal_index = 0 

        # Get anomaly samples and repeat based on available normal samples
        bad_traces_rep = np.repeat(bad_traces, self.samples_per_batch, 0)
        
        good_traces_rep = np.repeat(good_traces, self.samples_per_batch, 0)
        XR = np.concatenate([bad_traces_rep, good_traces_rep])
        
        bad_label = np.ones(bad_traces_rep.shape[0])
        good_label = np.zeros(good_traces_rep.shape[0])
        y = np.concatenate([bad_label, good_label])

        # Get random normal samples
        XL = np.concatenate([self.rdm_normal_traces, self.rdm_normal_traces])

        # Additional shuffle required
        new_indices = np.arange((y.shape[0]))
        np.random.shuffle(new_indices)
        XR = np.expand_dims(XR[new_indices], 2)
        XL = np.expand_dims(XL[new_indices], 2)
        y = y[new_indices]
        
        if self.mpsLabels != None:
            fault_labels = self.fault_ids[indexes]
            fault_labels_rep = np.repeat(fault_labels, self.samples_per_batch, 0)
            fault_labels_out = np.concatenate([fault_labels_rep, ["Normal"]*len(fault_labels_rep)])
            fault_labels_out = fault_labels_out[new_indices]
            return (XL, XR), y, fault_labels_out
        if self.decoder != None:   
            if self.decoder.lower() in ["left", "l"]:
                return (XL, XR), (y, XL)
            elif self.decoder.lower() in ["right", 'r']:
                return (XL, XR), (y, XR)    
        return (XL, XR), y

    def on_epoch_end(self):
        'Updates indexes after each epoch'

        self.anomaly_indices = np.arange(len(self.anomaly_traces))
        self.normal_indices = np.arange(len(self.normal_traces))
        if self.shuffle:
            np.random.shuffle(self.anomaly_indices)
            np.random.shuffle(self.normal_indices)
        
        
        self.normal_indexes = np.random.choice(self.normal_size - 1,
                                                size=self.samples_per_batch*self.num_anomalies_batch,
                                                replace=False,
                                                p=None)

        self.rdm_normal_traces = self.normal_traces[self.normal_indexes]
