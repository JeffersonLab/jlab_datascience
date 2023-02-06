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
import math

# Euclidean distance = sqrt(sum(square(t1-t2)))
def euclidean_distance(vects):
    """Find the Euclidean distance between two vectors.

    Arguments:
        vects: List containing two tensors of same length.

    Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """

    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))

class DeepResNet(tf.keras.Model):
  """Defines a multi-layer residual network."""
  def __init__(self, num_classes, num_layers=3, num_hidden=128,
               dropout_rate=0.1, **classifier_kwargs):
    super().__init__()
    # Defines class meta data.
    self.num_hidden = num_hidden
    self.num_layers = num_layers
    self.dropout_rate = dropout_rate
    self.classifier_kwargs = classifier_kwargs

    # Defines the hidden layers.
    self.input_layer = tf.keras.layers.Dense(self.num_hidden, trainable=False)
    self.dense_layers = [self.make_dense_layer() for _ in range(num_layers)]

    # Defines the output layer.
    self.classifier = self.make_output_layer(num_classes)

  def call(self, inputs):
    # Projects the 2d input data to high dimension.
    hidden = self.input_layer(inputs)

    # Computes the resnet hidden representations.
    for i in range(self.num_layers):
      resid = self.dense_layers[i](hidden)
      resid = tf.keras.layers.Dropout(self.dropout_rate)(resid)
      hidden += resid

    return self.classifier(hidden)

  def make_dense_layer(self):
    """Uses the Dense layer as the hidden layer."""
    return tf.keras.layers.Dense(self.num_hidden, activation="relu")

  def make_output_layer(self, num_classes):
    """Uses the Dense layer as the output layer."""
    return tf.keras.layers.Dense(
        num_classes, **self.classifier_kwargs)

class DeepResNetSNGP(DeepResNet):
  def __init__(self, spec_norm_bound=0.9, **kwargs):
    self.spec_norm_bound = spec_norm_bound
    super().__init__(**kwargs)

  def make_dense_layer(self):
    """Applies spectral normalization to the hidden layer."""
    dense_layer = super().make_dense_layer()
    return SpectralNormalization(
      dense_layer, norm_multiplier=self.spec_norm_bound)

  def make_output_layer(self, num_classes):
    """Uses Gaussian process as the output layer."""
    return RandomFeatureGaussianProcess(
      num_classes,
      gp_cov_momentum=-1,
      **self.classifier_kwargs)

  def call(self, inputs, training=False, return_covmat=True):
    # Gets logits and covariance matrix from GP layer.
    logits, covmat = super().call(inputs)

    # Returns only logits during training.
    if not training and return_covmat:
      return logits, covmat

    return logits


class SiameseModelSNGP_DenseOnly(tf.keras.Model):
    """"""
    def __init__(self, nClasses=2, spec_norm_bound=0.9,  num_layers=3, num_hidden=128,
               dropout_rate=0.1, distanceMetric="L1", **classifier_kwargs):
        self.spec_norm_bound = spec_norm_bound
        super().__init__()
        
        # Defines class meta data.
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.classifier_kwargs = classifier_kwargs
        self.nClasses = nClasses
        self.distanceMetric = distanceMetric
        
        # Denife ResNet Layers
        self.input_layer = tf.keras.layers.Dense(self.num_hidden, trainable=False)
        self.dense_layers = [self.make_dense_layer() for _ in range(num_layers)]
        
        # Define FewShot difference Layers
        self.L1_layer = tf.keras.layers.Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]), name='L1_distance')
        self.L2_layer = tf.keras.layers.Lambda(lambda tensors: K.square(tensors[0] - tensors[1]), name='L2_distance')
        
        # Define Output Layer
        self.classifier = self.make_output_layer()
        
    def call(self, inputs, training=False, return_covmat=False):
        """"""
        encodedL = self.callToDeepResNet(inputs[0])
        encodedR = self.callToDeepResNet(inputs[1])
        if self.distanceMetric.lower() == "l2":
            print("Using L2 distance.")
            distance = self.L2_layer([encodedL, encodedR])
        else:
            print("Using L1 distance.")
            distance = self.L1_layer([encodedL, encodedR])
        distance = tf.keras.layers.Dropout(0.2)(distance)
        logits, covmat = self.classifier(distance)
        if not training and return_covmat:
            return logits, covmat

        return logits
    
    def make_dense_layer(self):
        """Uses the Dense layer as the hidden layer."""
        dense_layer = tf.keras.layers.Dense(self.num_hidden, activation="relu")
        return SpectralNormalization(
            dense_layer, norm_multiplier=self.spec_norm_bound)

    def make_output_layer(self):
        """Uses the Dense layer as the output layer."""
#         return tf.keras.layers.Dense(
#             self.nClasses, activation="sigmoid")
        return RandomFeatureGaussianProcess(
                                        self.nClasses, 
                                        gp_cov_momentum=-1,
                                        **self.classifier_kwargs)
    def callToDeepResNet(self, inp):
        """"""
        hidden = self.input_layer(inp)
        for i in range(self.num_layers):
            resid = self.dense_layers[i](hidden)
            resid = tf.keras.layers.Dropout(self.dropout_rate)(resid)
            hidden += resid
        return hidden


class ResNet1D(tf.keras.Model):
    """ResNet Model with or without Spectral Normalization"""
    def __init__(self, input_shape=(10000, 1), spec_norm_bound=0.9, num_filters=[32, 64, 128], strides=[2,2,2], dropPercentage=0.05, UQ=False, decoder=False):
        
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
        # if not sesiameseModelWithResNetlf.UQ:
        self.BN_layers = [self.make_BN_layers() for _ in range(len(num_filters))]
        self.BN_layers_id = [self.make_BN_layers() for _ in range(len(num_filters))]
            
        self.maxpool_layers = [tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding="same") 
                               for _ in range(len(num_filters))]
        self.dense_layer = self.make_dense_layer(1, activation="relu")
        self.activation_layers = [tf.keras.layers.Activation("relu") for _ in range(len(num_filters))]
        self.dropout_layers = [tf.keras.layers.Dropout(self.dropPercentage) for _ in range(len(num_filters))]
        self.Add_layers = [tf.keras.layers.Add() for _ in range(len(num_filters))]
        self.decoder = decoder

        if self.decoder:
            self.make_decoder()
            
            decoded_length = input_shape[0]
            for s in strides:
                decoded_length = math.ceil(decoded_length / (s*2))
            for s in strides:
                decoded_length = decoded_length * s
            for c in range(len(self.up_sampling_layers)):
                decoded_length = decoded_length * 2
            print("Decoded length estimated to be: ", decoded_length)
            cropping_total = decoded_length - input_shape[0]
            if cropping_total % 2 == 0:
                self.beginning_cropping = self.end_cropping = int(cropping_total/2)
            else:
                self.beginning_cropping = int(cropping_total/2)
                self.end_cropping = int(cropping_total/2)+1
            print("cropping applied are: ", self.beginning_cropping, self.end_cropping)
            self.cropping_layer = tf.keras.layers.Cropping1D(cropping=(self.beginning_cropping, self.end_cropping))
    
    def call(self, inputs, training=False, return_last_conv=False):
        "Model call function"
        hidden = inputs
        for i in range(len(self.num_filters)):
            if i > 1:
                #Id Block
                id_out = self.id_layers[i-1](hidden)
                id_out = self.BN_layers_id[i - 1](id_out)
                # Add the output of id layer to the original input
                hidden = self.Add_layers[i-1]([hidden, id_out])
            
            # Conv block
            conv_out = self.conv_layers[i](hidden)
            conv_out = self.BN_layers[i](conv_out)
            conv_out = self.maxpool_layers[i](conv_out)
            conv_out = self.activation_layers[i](conv_out)
            hidden = self.dropout_layers[i](conv_out)
        out = self.dense_layer(hidden)
        embedding = tf.keras.layers.Flatten()(out)
        if self.decoder:
            decoded_out = self.dense_layer_decode(out)
            for i in range(len(self.conv_layers_decode)-1):
                decoded_out = self.up_sampling_layers[i](decoded_out)
                decoded_out = self.conv_layers_decode[i](decoded_out)
                decoded_out = self.id_layers_decode[i](decoded_out)
            decoded_out = self.conv_layers_decode[-1](decoded_out)
            decoded_out = self.cropping_layer(decoded_out)
            if return_last_conv:
                return embedding, decoded_out, conv_out
            return embedding, decoded_out
        if return_last_conv:
            return embedding, conv_out
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
    
    def make_decoder(self):
        """ Makes layers for decoder of the same network, can be turned of by passing the argument Decoder=False
        """
        n_filters = len(self.num_filters)
        self.conv_layers_decode = []
        self.up_sampling_layers = []
        self.id_layers_decode = []
        for i in range(len(self.num_filters)):
            # Add two reverse conv layers, one for conv and one to offset pooling
            self.up_sampling_layers.append(tf.keras.layers.UpSampling1D(size=2))
            self.conv_layers_decode.append(tf.keras.layers.Conv1DTranspose(filters=self.num_filters[n_filters-i-1], kernel_size=3, strides=self.strides[n_filters-i-1], padding="same", activation="relu"))
            self.id_layers_decode.append(tf.keras.layers.Conv1DTranspose(filters=self.num_filters[n_filters-i-1], kernel_size=3, strides=1, padding="same", activation="relu"))
        self.conv_layers_decode.append(tf.keras.layers.Conv1D(filters=1, kernel_size=3, padding="same", activation="sigmoid"))
        print(len(self.conv_layers_decode))
        self.dense_layer_decode = self.make_dense_layer(self.num_filters[-1], activation="relu")



class siameseModelWithResNet(ResNet1D):
    """
    """
    def __init__(self, input_shape=(10000, 1), spec_norm_bound=0.9, num_filters=[32, 64, 128], 
                 strides=[2,2,2], dropPercentage=0.05, UQ=False, 
                distanceMetric="l2", CommonDense_nodes=[64], CommonDrop_percentage=[0.15], nClasses=1, lambdaShift=-10, decoder=None,
                 **classifier_kwargs):
        self.decoder_flag = False
        if decoder != None:
            self.decoder_flag = True
        super().__init__(input_shape=input_shape, spec_norm_bound=spec_norm_bound, num_filters=num_filters, 
                 strides=strides, dropPercentage=dropPercentage, UQ=UQ, decoder=self.decoder_flag)
        
        self.UQ = UQ
        self.nClasses = nClasses
        self.distanceMetric = distanceMetric
        self.classifier_kwargs = classifier_kwargs
        self.decoder = decoder
        self.L1_layer = tf.keras.layers.Lambda(lambda tensors: K.abs(tensors[0] - tensors[1])+lambdaShift, 
                                               name='L1_distance')
        self.L2_layer = tf.keras.layers.Lambda(lambda tensors: K.abs(K.square(tensors[0]) - K.square(tensors[1]))+lambdaShift, 
                                               name='L2_distance')
        self.CommonDense_layers = [self.make_dense_layer(n, activation="linear")
                                  for n in CommonDense_nodes]
        self.CommonDrop_layers = [tf.keras.layers.Dropout(d) for d in CommonDrop_percentage]
        self.classifier = self.make_output_layer()
        
    def call(self, inputs, training=False, return_covmat=False, return_conv=False, return_latent=False, return_decoded=False):
        """"""
        left_output = super().call(inputs[0], return_last_conv=return_conv)
        right_output = super().call(inputs[1], return_last_conv=return_conv)
        if not self.decoder_flag and not return_conv:
            encodedL = left_output
            encodedR = right_output
            decodedL = decodedR = convL = convR = None
        elif self.decoder_flag and not return_conv:
            encodedL, decodedL = left_output
            encodedR, decodedR = right_output
            convL = convR = None
        elif self.decoder_flag and return_conv:
            encodedL, decodedL, convL = left_output
            encodedR, decodedR, convR = right_output

        if self.distanceMetric.lower() == "l2":
            distance = self.L2_layer([encodedL, encodedR])
        else:
            distance = self.L1_layer([encodedL, encodedR])
        hidden = tf.keras.layers.Dropout(0.2)(distance)

        for i in range(len(self.CommonDense_layers)):
            hidden = self.CommonDense_layers[i](hidden)
            hidden = self.CommonDrop_layers[i](hidden)
            
        if self.UQ:
            logits, covmat = self.classifier(hidden)
        else:
            logits = self.classifier(hidden)
            covmat = None
        
        # if training, and decoder=left/right: return logits and decodedL/decodedR
        # if not training: return logits and a tuple of other outputs as (covmat, convL, convR, decodedL, decodedR, distance)
        # tuple includes covmat if UQ=True else None
        # tuple includes convL and convR if return_conv=True else None, None
        # tuple includes decodedL, and decodedR if self.decoder_flag=True else None, None
        # tuple includes distance if return_latent=True else None
        if self.decoder_flag:
            if self.decoder.lower() in ["left", "l"]:
                decoded_output = decodedL
            else:
                decoded_output = decodedR

        if training and self.decoder_flag:
            return logits, decoded_output
        elif training and not self.decoder_flag:
            return logits
        else:
            dictionary = {"covmat":covmat, 
                          "convL": convL, 
                          "convR": convR, 
                          "decodedL": decodedL, 
                          "decodedR": decodedR, 
                          "latent": distance}
            if return_covmat or return_conv or return_latent or return_decoded:
                return logits, dictionary
            elif self.decoder_flag:
                return logits, decoded_output
            return logits
            
    def make_dense_layer(self, nodes, activation="relu"):
        """Uses the Dense layer as the hidden layer."""
        return tf.keras.layers.Dense(nodes, activation=activation)
        
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