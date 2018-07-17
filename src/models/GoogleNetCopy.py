from tensorflow.python.keras.layers import ( Input, Dense, Convolution2D, MaxPooling2D )
from tensorflow.python.keras.layers import ( AveragePooling2D, ZeroPadding2D, Dropout, Flatten ) 
from tensorflow.python.keras.layers import ( Concatenate, Reshape, Activation, BatchNormalization )
from tensorflow.python.keras.models import Model

from layers import PoolHelper, LRN

class GoogLeNetContext( object ):
    '''
    Class encapsulates the input to the GoogleNet class to reduce constructor argument count.
    '''
    def __init__( self, activation = 'relu', input_shape = ( 224, 224, 3 ), use_batchnorm = False ):
        '''
        Constructor for the Initialization of the class
        '''
        self.__activation    = activation
        self.__input_shape   = input_shape 
        self.__use_batchnorm = use_batchnorm

    @property
    def activation( self ):
        '''
        Getter for the Activation Type
        '''
        return self.__activation

    @property
    def input_shape( self ):
        '''
        Getter for the Input Shape
        '''
        return self.__input_shape

    @property
    def use_batchnorm( self ):
        '''
        Getter for the flag to use batch norm
        '''
        return self.__use_batchnorm


class GoogLeNet( object ):
    ''' 
    This class is responsible for the creation of the Google Net Architecture. 
    '''

    def __init__( self, context ): 
        '''
        '''
        self.__activation  = context.activation 
        self.__input_shape = context.input_shape 
        self.__model       = self.__initialize_model( self.__input_shape, self.__activation )

    def __initialize_model( self, input_shape, activation ):
        '''
        '''
        input_layer = Input( shape=input_shape )

        conv1_7x7_s2 = Convolution2D(64,
                                     (7, 7),
                                     strides=(2, 2),
                                     padding='same',
                                     activation=activation,
                                     name='conv1/7x7_s2')( input_layer )

        conv1_zero_pad = ZeroPadding2D(padding=(1, 1))(conv1_7x7_s2)

        pool1_helper = PoolHelper()(conv1_zero_pad)

        pool1_3x3_s2 = MaxPooling2D( pool_size=(3, 3), 
                                     strides=(2, 2), 
                                     padding='valid', 
                                     name='pool1/3x3_s2' )(pool1_helper)

        pool1_norm1 = LRN(name='pool1/norm1', batch_size=32)(pool1_3x3_s2)

        conv2_3x3_reduce = Convolution2D(64, 
                                         (1, 1), 
                                         padding='same', 
                                         activation=activation, 
                                         name='conv2/3x3_reduce')(pool1_norm1)

        conv2_3x3 = Convolution2D(192, 
                                  (3, 3), 
                                  padding='same', 
                                  activation=activation, 
                                  name='conv2/3x3')(conv2_3x3_reduce)

        conv2_norm2 = LRN(name='conv2/norm2')(conv2_3x3)

        conv2_zero_pad = ZeroPadding2D(padding=(1, 1))(conv2_norm2)

        pool2_helper = PoolHelper()(conv2_zero_pad)

        pool2_3x3_s2 = MaxPooling2D(pool_size=(3, 3), 
                                    strides=(2, 2), 
                                    padding='valid', 
                                    name='pool2/3x3_s2')(pool2_helper)

        filters_3a = {'1x1': 64, '3x3_reduce': 96, '3x3': 128, '5x5_reduce': 16, '5x5': 32, 'pool_proj': 32}

        inception_3a_output = self.__inception_module(name='3a',
                                                      pooled_input=pool2_3x3_s2,
                                                      filters=filters_3a,
                                                      activation=activation)

        filters_3a = {'1x1': 64, '3x3_reduce': 96, '3x3': 128, '5x5_reduce': 16, '5x5': 32, 'pool_proj': 32}

        inception_3a_output = self.__inception_module(name='3a',
                                                      pooled_input=inception_3a_output,
                                                      filters=filters_3a,
                                                      activation=activation)

        filters_3b = {'1x1': 128, '3x3_reduce': 128, '3x3': 192, '5x5_reduce': 32, '5x5': 96, 'pool_proj': 64}

        inception_3b_output = self.__inception_module(name='3b',
                                                      pooled_input=inception_3a_output,
                                                      filters=filters_3b,
                                                      activation=activation)

        inception_3b_output_zero_pad = ZeroPadding2D(padding=(1, 1))(inception_3b_output)

        pool3_helper = PoolHelper()(inception_3b_output_zero_pad)

        pool3_3x3_s2 = MaxPooling2D(pool_size=(3, 3), 
                                    strides=(2, 2), 
                                    padding='valid', 
                                    name='pool3/3x3_s2')(pool3_helper)

        filters_4a = {'1x1': 192, '3x3_reduce': 96, '3x3': 208, '5x5_reduce': 16, '5x5': 48, 'pool_proj': 64}

        inception_4a_output = self.__inception_module(name='4a',
                                                      pooled_input=pool3_3x3_s2,
                                                      filters=filters_4a,
                                                      activation=activation)

        loss1_ave_pool = AveragePooling2D(pool_size=(5, 5), 
                                          strides=(3, 3), 
                                          name='loss1/ave_pool')(inception_4a_output)

        loss1_conv = Convolution2D(128, 
                                   (1, 1), 
                                   padding='same', 
                                   activation=activation, 
                                   name='loss1/conv')(loss1_ave_pool)

        loss1_flat = Flatten()(loss1_conv)

        loss1_fc = Dense(1024, activation=activation, name='loss1/fc')(loss1_flat)

        loss1_drop_fc = Dropout(0.7)(loss1_fc)

        loss1_classifier = Dense(1000, name='loss1/classifier')(loss1_drop_fc)

        loss1_classifier_act = Activation('softmax')(loss1_classifier)

        filters_4b = {'1x1': 160, '3x3_reduce': 112, '3x3': 224, '5x5_reduce': 24, '5x5': 64, 'pool_proj': 64}

        inception_4b_output = self.__inception_module(name='4b',
                                                      pooled_input=inception_4a_output,
                                                      filters=filters_4b,
                                                      activation=activation)

        filters_4c = {'1x1': 128, '3x3_reduce': 128, '3x3': 256, '5x5_reduce': 24, '5x5': 64, 'pool_proj': 64}

        inception_4c_output = self.__inception_module(name='4c',
                                                      pooled_input=inception_4b_output,
                                                      filters=filters_4c,
                                                      activation=activation)

        filters_4d = {'1x1': 112, '3x3_reduce': 144, '3x3': 288, '5x5_reduce': 32, '5x5': 64, 'pool_proj': 64}

        inception_4d_output = self.__inception_module(name='4d',
                                                      pooled_input=inception_4c_output,
                                                      filters=filters_4d,
                                                      activation=activation)

        loss2_ave_pool = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), name='loss2/ave_pool')(inception_4d_output)

        loss2_conv = Convolution2D(128, 1, 1, padding='same', activation=activation, name='loss2/conv')(loss2_ave_pool)

        loss2_flat = Flatten()(loss2_conv)

        loss2_fc = Dense(1024, activation=activation, name='loss2/fc')(loss2_flat)

        loss2_drop_fc = Dropout(0.7)(loss2_fc)

        loss2_classifier = Dense(1000, name='loss2/classifier')(loss2_drop_fc)

        loss2_classifier_act = Activation('softmax')(loss2_classifier)

        filters_4e = {'1x1': 256, '3x3_reduce': 160, '3x3': 320, '5x5_reduce': 32, '5x5': 128, 'pool_proj': 128}

        inception_4e_output = self.__inception_module(name='4e',
                                                      pooled_input=inception_4d_output,
                                                      filters=filters_4e,
                                                      activation=activation)

        inception_4e_output_zero_pad = ZeroPadding2D(padding=(1, 1))(inception_4e_output)

        pool4_helper = PoolHelper()(inception_4e_output_zero_pad)

        pool4_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='pool4/3x3_s2')(pool4_helper)

        filters_5a = {'1x1': 256, '3x3_reduce': 160, '3x3': 320, '5x5_reduce': 32, '5x5': 128, 'pool_proj': 128}

        inception_5a_output = self.__inception_module(name='5a',
                                                      pooled_input=pool4_3x3_s2,
                                                      filters=filters_5a,
                                                      activation=activation)

        filters_5b = {'1x1': 384, '3x3_reduce': 192, '3x3': 384, '5x5_reduce': 48, '5x5': 128, 'pool_proj': 128}

        inception_5b_output = self.__inception_module(name='5b',
                                                      pooled_input=inception_5a_output,
                                                      filters=filters_5b,
                                                      activation=activation)

        pool5_7x7_s1 = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), name='pool5/7x7_s2')(inception_5b_output)

        loss3_flat = Flatten()(pool5_7x7_s1)

        pool5_drop_7x7_s1 = Dropout(0.4)(loss3_flat)

        loss3_classifier = Dense(1000, name='loss3/classifier')(pool5_drop_7x7_s1)

        loss3_classifier_act = Activation('softmax', name='prob')(loss3_classifier)

        model = Model(inputs= input_layer , outputs=[loss1_classifier_act, loss2_classifier_act, loss3_classifier_act])

        return model

    def __convolution_layer_with_batchnorm( self, 
                                            x,
                                            filters,
                                            kernel_size,
                                            padding='same',
                                            strides=(1, 1),
                                            activation='relu',
                                            before_activation=False,
                                            name=None ):
        x = Convolution2D(filters,
                          kernel_size,
                          strides=strides,
                          padding=padding,
                          use_bias=False,
                          name=name)(x)

        if before_activation:
            x = BatchNormalization(axis=-1, scale=False, name=name+'/bn')(x)
            x = Activation(activation, name=name+'/act')(x)
        else:
            x = Activation(activation, name=name+'/act')(x)
            x = BatchNormalization(axis=-1, scale=False, name=name+'/bn')(x)
        return x


    def __inception_module(self, name, pooled_input, filters):
        inception_1x1 = Convolution2D(filters['1x1'],
                                      (1, 1),
                                      padding='same',
                                      activation=self.__activation,
                                      name='inception_' + name + '/1x1')(pooled_input)

        inception_3x3_reduce = Convolution2D(filters['3x3_reduce'],
                                             (1, 1),
                                             padding='same',
                                             activation=self.__activation,
                                             name='inception_' + name + '/3x3_reduce')(pooled_input)

        inception_3x3 = Convolution2D(filters['3x3'],
                                      (3, 3),
                                      padding='same',
                                      activation=self.__activation,
                                      name='inception_' + name + '/3x3')(inception_3x3_reduce)

        inception_5x5_reduce = Convolution2D(filters['5x5_reduce'],
                                             (1, 1),
                                             padding='same',
                                             activation=self.__activation,
                                             name='inception_' + name + '/5x5_reduce')(pooled_input)

        inception_5x5 = Convolution2D(filters['5x5'],
                                      (5, 5),
                                      padding='same',
                                      activation=self.__activation,
                                      name='inception_' + name + '/5x5')(inception_5x5_reduce)

        inception_pool = MaxPooling2D(pool_size=(3, 3),
                                      strides=(1, 1),
                                      padding='same',
                                      name='inception_' + name + '/pool')(pooled_input)

        inception_pool_proj = Convolution2D(filters['pool_proj'],
                                            (1, 1),
                                            padding='same',
                                            activation=self.__activation,
                                            name='inception_' + name + '/pool_proj')(inception_pool)

        inception_output = Concatenate(name='inception_' + name + '/output')([inception_1x1,
                                                                              inception_3x3,
                                                                              inception_5x5,
                                                                              inception_pool_proj])
        return inception_output
