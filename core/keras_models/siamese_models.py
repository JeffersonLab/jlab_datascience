import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
try:
    from official.nlp.modeling.layers.spectral_normalization import SpectralNormalization
    from official.nlp.modeling.layers.gaussian_process import RandomFeatureGaussianProcess
    import tensorflow_addons as tfa
    import official.nlp.modeling.layers as nlp_layers
except:
    print("Not able to import tf official packages for UQ, skipping...")


class ResNet1D(tf.keras.Model):
    """ResNet Model with or without Spectral Normalization"""
    def __init__(self, input_shape=(10000, 1), spec_norm_bound=0.9, num_filters=[32, 64, 128], strides=[2,2,2], dropPercentage=0.05, UQ=False):
        
        if len(num_filters) != len(strides):
            print("Error!!! Arrays containing Number of filters and Strides information should be of same length.")
            return
    
        self.spec_norm_bound = spec_norm_bound
        super().__init__()
        self.num_filters = num_filters
        self.inp_shape = input_shape
        self.UQ = UQ
        self.strides=strides
        self.dropPercentage = dropPercentage
        self.input_layer = tf.keras.layers.Input(shape=self.inp_shape)
        self.id_layers = [self.make_conv1d(filters=self.num_filters[i], k_size=(1), strides=(1))
                         for i in range(len(num_filters))]
        
        self.conv_layers = [self.make_conv1d(filters=self.num_filters[i], strides=self.strides[i])
                           for i in range(len(num_filters))]
        self.BN_layers = [self.make_BN_layers() for _ in range(len(num_filters))]
        self.BN_layers_id = [self.make_BN_layers() for _ in range(len(num_filters))]
            
        self.maxpool_layers = [tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding="same") 
                               for _ in range(len(num_filters))]
        self.dense_layer = self.make_dense_layer(1, activation="relu")
        self.activation_layers = [tf.keras.layers.Activation("relu") for _ in range(len(num_filters))]
        self.dropout_layers = [tf.keras.layers.Dropout(self.dropPercentage) for _ in range(len(num_filters))]
        self.Add_layers = [tf.keras.layers.Add() for _ in range(len(num_filters))]
    
    def call(self, inputs, training=False):
        "Model call function"
        hidden = inputs
        for i in range(len(self.num_filters)):
            if i > 1:
                #Id Block
                id_out = self.id_layers[i-1](hidden)
                id_out = self.BN_layers_id[i - 1](id_out)
                hidden = self.Add_layers[i-1]([hidden, id_out])
            
            # Conv block
            conv_out = self.conv_layers[i](hidden)
            conv_out = self.BN_layers[i](conv_out)
            conv_out = self.maxpool_layers[i](conv_out)
            conv_out = self.activation_layers[i](conv_out)
            hidden = self.dropout_layers[i](conv_out)
        out = self.dense_layer(hidden)
        embedding = tf.keras.layers.Flatten()(out)
        return embedding
        
        
    def make_dense_layer(self, nodes, activation="relu"):
        """Uses the Dense layer as the hidden layer."""
        dense_layer = tf.keras.layers.Dense(nodes, activation=activation)
        return dense_layer


    def make_conv1d(self, filters=16, k_size=3, strides=2, activation="relu", padding="same"):
        """Make Conv1D layers and use as hidden layer"""
        conv_layer = keras.layers.Conv1D(filters, 
                                         kernel_size=k_size, 
                                         strides=strides, 
                                         activation=activation, 
                                         padding=padding)
        return conv_layer
        
    def make_BN_layers(self):
        """Make Batch Normalization layer"""
        BN_layer = keras.layers.BatchNormalization()
        return BN_layer
    
    def summary(self):
        """"""
        return tf.keras.models.Model(inputs=self.input_layer, outputs=self.call(self.input_layer)).summary()
        
    
    def plot(self, save_loc=""):
        """ """
        return tf.keras.utils.plot_model(
            tf.keras.models.Model(inputs=self.input_layer, outputs=self.call(self.input_layer)),                      # here is the trick (for now)
            dpi=96, 
            show_shapes=True, show_layer_names=True,  # show shapes and layer name
            expand_nested=False                       # will show nested block
        )


class SiameseModelWithResNet(ResNet1D):
    """
    """
    def __init__(self, input_shape=(10000, 1), spec_norm_bound=0.9, num_filters=[32, 64, 128], 
                 strides=[2,2,2], dropPercentage=0.05, UQ=False,
                distanceMetric="l2", CommonDense_nodes=[64], CommonDrop_percentage=[0.15], nClasses=1, lambdaShift=-10,
                 **classifier_kwargs):
        super().__init__(input_shape=input_shape, spec_norm_bound=spec_norm_bound, num_filters=num_filters, 
                 strides=strides, dropPercentage=dropPercentage, UQ=UQ)
        self.UQ = UQ
        self.nClasses = nClasses
        self.distanceMetric = distanceMetric
        self.classifier_kwargs = classifier_kwargs
        self.L1_layer = tf.keras.layers.Lambda(lambda tensors: K.abs(tensors[0] - tensors[1])+lambdaShift, 
                                               name='L1_distance')
        self.L2_layer = tf.keras.layers.Lambda(lambda tensors: K.abs(K.square(tensors[0]) - K.square(tensors[1]))+lambdaShift, 
                                               name='L2_distance')
        self.CommonDense_layers = [self.make_dense_layer(n, activation="linear")
                                  for n in CommonDense_nodes]
        self.CommonDrop_layers = [tf.keras.layers.Dropout(d) for d in CommonDrop_percentage]
        self.classifier = self.make_output_layer()
        
    def call(self, inputs, training=False, return_covmat=True):
        """"""
        encodedL = super().call(inputs[0])
        encodedR = super().call(inputs[1])

        if self.distanceMetric.lower() == "l2":
            distance = self.L2_layer([encodedL, encodedR])
        else:
            distance = self.L1_layer([encodedL, encodedR])
        hidden = tf.keras.layers.Dropout(0.2)(distance)

        for i in range(len(self.CommonDense_layers)):
            hidden = self.CommonDense_layers[i](hidden)
            hidden = tf.keras.layers.LeakyReLU()(hidden)
            hidden = self.CommonDrop_layers[i](hidden)
        if self.UQ:
            logits, covmat = self.classifier(hidden)
            if not training and return_covmat:
                return logits, covmat
            else:
                return logits
        else:
            logits = self.classifier(hidden)
            return logits
            
    def make_dense_layer(self, nodes, activation="relu"):
        """Uses the Dense layer as the hidden layer."""
        dense_layer = tf.keras.layers.Dense(nodes, activation=activation)
        return dense_layer
        
    def make_output_layer(self):
        """Uses the Dense layer as the output layer."""
        if self.UQ:
            return nlp_layers.RandomFeatureGaussianProcess(
                                        self.nClasses, 
                                        gp_cov_momentum=-1,
                                        custom_random_features_activation=tf.nn.sigmoid,
                                        **self.classifier_kwargs)
        else:
            return tf.keras.layers.Dense(self.nClasses, activation="sigmoid")
        
    def summary(self):
        x = tf.keras.layers.Input(self.inp_shape)
        y = tf.keras.layers.Input(self.inp_shape)
        return tf.keras.models.Model(inputs=[x, y], outputs=self.call([x, y])).summary()
    
    def plot(self):
        x = tf.keras.layers.Input(self.inp_shape)
        y = tf.keras.layers.Input(self.inp_shape)
        return tf.keras.utils.plot_model(
            tf.keras.models.Model(inputs=[x, y], outputs=self.call([x, y])),     # here is the trick (for now)
            dpi=96, 
            show_shapes=True, show_layer_names=True,  # show shapes and layer name
            expand_nested=False                       # will show nested block
        )