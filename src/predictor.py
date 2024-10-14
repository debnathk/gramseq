from __future__ import print_function
from __future__ import division
import keras
from keras import backend as K
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau
import molecule_vae

PATH = '/home/debnathk/gramseq/'

def sampling(args):
    z_mean_, z_log_var_ = args
    batch_size = K.shape(z_mean_)[0]
    epsilon = K.random_normal(shape=(batch_size, 56), mean=0., stddev = 0.01)
    return z_mean_ + K.exp(z_log_var_ / 2) * epsilon

class DLEPS(object):
    def __init__(self, setmean = False, reverse = True, base = -2, up_name=None, down_name=None, save_exp = None, rnaseq=False, protenc=None):

        
        self.save_exp = save_exp
        self.reverse = reverse
        self.base = base
        self.setmean = setmean
        self.loaded = False

        self.rnaseq = rnaseq
        self.protenc = protenc
        
        self.model = []
        self.model.append(self._build_model())

    def _build_model(self):
        
        # gVAE network - for drugs
        grammar_weights = PATH + 'data/vae.hdf5'
        grammar_model = molecule_vae.ZincGrammarModel(grammar_weights)
        grammar_model.trainable = False
        self.grammar_model = grammar_model
        z_mn, z_var = grammar_model.vae.encoderMV.output
        latent_dim = 56
        output_gvae = keras.layers.Lambda(sampling, output_shape=(latent_dim,), name='lambda')([z_mn, z_var])

        # Dense network - for rna seq data
        input_dense = keras.Input(shape=(978, 2))
        x = keras.layers.Flatten()(input_dense)
        x = keras.layers.Dense(256, activation="relu")(x)
        x = keras.layers.Dropout(0.4)(x)
        x = keras.layers.Dense(128, activation="relu")(x)
        x = keras.layers.Dropout(0.4)(x)
        output_dense = keras.layers.Dense(latent_dim, activation="softmax")(x)

        # RNN (LSTM) Model - for proteins
        '''Define the RNN encoder for target'''

        if self.protenc == 'RNN':
        # Create layers
            visible_1 = keras.Input(shape=(26, 1000))
            x = visible_1
            x = keras.layers.LSTM(units=128, return_sequences=True)(visible_1)
            x = keras.layers.LSTM(units=64)(x)
            x = keras.layers.Dense(latent_dim, activation="relu")(x)
            output_rnn = keras.layers.Dense(latent_dim, activation="softmax")(x)

            # Concatenate layers
            if self.rnaseq:
                merge = keras.layers.concatenate([output_gvae, output_dense, output_rnn])
                x = keras.layers.Dense(1024,activation='relu', kernel_regularizer=keras.regularizers.l2(0.00001))(merge)
                x = keras.layers.Dropout(0.4)(x)
                x = keras.layers.Dense(1024,activation='relu', kernel_regularizer=keras.regularizers.l2(0.00001))(x)
                x = keras.layers.Dropout(0.4)(x)
                x = keras.layers.Dense(512,activation='relu', kernel_regularizer=keras.regularizers.l2(0.00001))(x)
                x = keras.layers.Dropout(0.4)(x)
                outputs = keras.layers.Dense(1, activation='linear')(x)
                model = keras.models.Model(inputs=[grammar_model.vae.encoderMV.input, input_dense, visible_1], outputs = outputs)
            else:
                merge = keras.layers.concatenate([output_gvae, output_rnn])
                x = keras.layers.Dense(1024,activation='relu', kernel_regularizer=keras.regularizers.l2(0.00001))(merge)
                x = keras.layers.Dropout(0.4)(x)
                x = keras.layers.Dense(1024,activation='relu', kernel_regularizer=keras.regularizers.l2(0.00001))(x)
                x = keras.layers.Dropout(0.4)(x)
                x = keras.layers.Dense(512,activation='relu', kernel_regularizer=keras.regularizers.l2(0.00001))(x)
                x = keras.layers.Dropout(0.4)(x)
                outputs = keras.layers.Dense(1, activation='linear')(x)
                model = keras.models.Model(inputs=[grammar_model.vae.encoderMV.input, visible_1], outputs = outputs)

        elif self.protenc == 'CNN':
            in_channels = [26, 32, 64, 96]
            kernels = [4, 8, 12]

            # Create layers
            visible_1 = keras.Input(shape=(26, 1000))
            x = visible_1
            for i in range(len(kernels)):
                x = keras.layers.Conv1D(filters=in_channels[i+1], kernel_size=kernels[i], activation="relu", padding="same")(x)
                x = keras.layers.MaxPooling1D(pool_size=2, padding="same")(x)
            x = keras.layers.GlobalMaxPooling1D()(x)
            x = keras.layers.Dense(latent_dim, activation="relu")(x)
            output_cnn = keras.layers.Dense(latent_dim, activation="softmax")(x)

            # Merge encoders
            if self.rnaseq:
                merge = keras.layers.concatenate([output_gvae, output_dense, output_cnn])
                x = keras.layers.Dense(1024,activation='relu', kernel_regularizer=keras.regularizers.l2(0.00001))(merge)
                x = keras.layers.Dropout(0.4)(x)
                x = keras.layers.Dense(1024,activation='relu', kernel_regularizer=keras.regularizers.l2(0.00001))(x)
                x = keras.layers.Dropout(0.4)(x)
                x = keras.layers.Dense(512,activation='relu', kernel_regularizer=keras.regularizers.l2(0.00001))(x)
                x = keras.layers.Dropout(0.4)(x)
                outputs = keras.layers.Dense(1, activation='linear')(x)
                model = keras.models.Model(inputs=[grammar_model.vae.encoderMV.input, input_dense, visible_1], outputs = outputs)
            else:
                merge = keras.layers.concatenate([output_gvae, output_cnn])
                x = keras.layers.Dense(1024,activation='relu', kernel_regularizer=keras.regularizers.l2(0.00001))(merge)
                x = keras.layers.Dropout(0.4)(x)
                x = keras.layers.Dense(1024,activation='relu', kernel_regularizer=keras.regularizers.l2(0.00001))(x)
                x = keras.layers.Dropout(0.4)(x)
                x = keras.layers.Dense(512,activation='relu', kernel_regularizer=keras.regularizers.l2(0.00001))(x)
                x = keras.layers.Dropout(0.4)(x)
                outputs = keras.layers.Dense(1, activation='linear')(x)
                model = keras.models.Model(inputs=[grammar_model.vae.encoderMV.input, visible_1], outputs = outputs)
        else:
            print('Choose between the following protein encodings - CNN, RNN')
            
        return model
    
    def train(self, smile_train, rna_train, validation_data, epochs=30000, batch_size=512, shuffle=True):
    
        assert (not self.loaded), 'Dense Model should not be loaded before training.'
        
        reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                                  factor = 0.2,
                                  patience = 10,
                                  min_lr = 0.0001)
        
        for layer in self.grammar_model.vae.encoderMV.layers:
            layer.trainable = False
        
        sgd = optimizers.SGD(lr=0.1, decay=0, momentum=0.9, nesterov=True)
        rms = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
        ada = optimizers.Adagrad(lr=0.001, epsilon=None, decay=0.0)
        adaD = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
        adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0)
        
        self.model[0].compile(optimizer=optimizers.Adagrad(lr=0.0002), loss='mean_squared_error')
        
        his = self.model[0].fit(smile_train, 
                rna_train, epochs=epochs,
                batch_size=batch_size,
                shuffle=shuffle, 
                validation_data=validation_data, # validation_data = (smile_val, rna_val)
                callbacks = [reduce_lr])
        
        return his