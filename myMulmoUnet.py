import tensorflow as tf


class MulmoUNet(object):
    def __init__(self, input_channel_count, output_channel_count, first_layer_filter_count):
        self.INPUT_IMAGE_SIZE = 256

        # encoder parameters
        self.CONV_FILTER_SIZE = 3
        self.CONV_STRIDE = 1
        self.CONV_PADDING = (1, 1)
        self.MAX_POOLING_SIZE = 3
        self.MAX_POOLING_STRIDE = 2
        self.MAX_POOLING_PADDING = (1, 1)

        # decoder parameters
        self.DECONV_FILTER_SIZE = 2
        self.DECONV_STRIDE = 2

        # -------------------------------------------ENCODER PART ------------------------------------------------------
        # (N x N x input_channel_count)
        input1 = tf.keras.Input(shape=(self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE, input_channel_count))
        input2 = tf.keras.Input(shape=(self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE, input_channel_count))
        input3 = tf.keras.Input(shape=(self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE, input_channel_count))
        input4 = tf.keras.Input(shape=(self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE, input_channel_count))
        input5 = tf.keras.Input(shape=(self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE, input_channel_count))

        # ENCODER 1
        enc0_1 = self._add_convolution_block(first_layer_filter_count, input1, 'LeakyReLU')
        enc0_1 = self._add_convolution_block(first_layer_filter_count, enc0_1, 'LeakyReLU')
        enc1_1 = self._add_encoding_block(first_layer_filter_count * 2, enc0_1)    # (N/2 x N/2 x 2CH)
        enc2_1 = self._add_encoding_block(first_layer_filter_count * 4, enc1_1)    # (N/4 x N/4 x 4CH)
        enc3_1 = self._add_encoding_block(first_layer_filter_count * 8, enc2_1)    # (N/8 x N/8 x 8CH)

        # ENCODER 2
        enc0_2 = self._add_convolution_block(first_layer_filter_count, input2, 'LeakyReLU')
        enc0_2 = self._add_convolution_block(first_layer_filter_count, enc0_2, 'LeakyReLU')
        enc1_2 = self._add_encoding_block(first_layer_filter_count * 2, enc0_2)    # (N/2 x N/2 x 2CH)
        enc2_2 = self._add_encoding_block(first_layer_filter_count * 4, enc1_2)    # (N/4 x N/4 x 4CH)
        enc3_2 = self._add_encoding_block(first_layer_filter_count * 8, enc2_2)    # (N/8 x N/8 x 8CH)

        # ENCODER 3
        enc0_3 = self._add_convolution_block(first_layer_filter_count, input3, 'LeakyReLU')
        enc0_3 = self._add_convolution_block(first_layer_filter_count, enc0_3, 'LeakyReLU')
        enc1_3 = self._add_encoding_block(first_layer_filter_count * 2, enc0_3)    # (N/2 x N/2 x 2CH)
        enc2_3 = self._add_encoding_block(first_layer_filter_count * 4, enc1_3)    # (N/4 x N/4 x 4CH)
        enc3_3 = self._add_encoding_block(first_layer_filter_count * 8, enc2_3)    # (N/8 x N/8 x 8CH)

        # ENCODER 4
        enc0_4 = self._add_convolution_block(first_layer_filter_count, input4, 'LeakyReLU')
        enc0_4 = self._add_convolution_block(first_layer_filter_count, enc0_4, 'LeakyReLU')
        enc1_4 = self._add_encoding_block(first_layer_filter_count * 2, enc0_4)    # (N/2 x N/2 x 2CH)
        enc2_4 = self._add_encoding_block(first_layer_filter_count * 4, enc1_4)    # (N/4 x N/4 x 4CH)
        enc3_4 = self._add_encoding_block(first_layer_filter_count * 8, enc2_4)    # (N/8 x N/8 x 8CH)

        # ENCODER 5
        enc0_5 = self._add_convolution_block(first_layer_filter_count, input5, 'LeakyReLU')
        enc0_5 = self._add_convolution_block(first_layer_filter_count, enc0_5, 'LeakyReLU')
        enc1_5 = self._add_encoding_block(first_layer_filter_count * 2, enc0_5)    # (N/2 x N/2 x 2CH)
        enc2_5 = self._add_encoding_block(first_layer_filter_count * 4, enc1_5)    # (N/4 x N/4 x 4CH)
        enc3_5 = self._add_encoding_block(first_layer_filter_count * 8, enc2_5)    # (N/8 x N/8 x 8CH)

        # -------------------------------------------- SHARED LATENT  -------------------------------------------------
        # max pooling 3 encoders (N/16 x N/16 x 8CH)
        latent_1 = tf.keras.layers.MaxPooling2D(pool_size=self.MAX_POOLING_SIZE, strides=self.MAX_POOLING_STRIDE,
                                                padding='same')(enc3_1)
        latent_2 = tf.keras.layers.MaxPooling2D(pool_size=self.MAX_POOLING_SIZE, strides=self.MAX_POOLING_STRIDE,
                                                padding='same')(enc3_2)
        latent_3 = tf.keras.layers.MaxPooling2D(pool_size=self.MAX_POOLING_SIZE, strides=self.MAX_POOLING_STRIDE,
                                                padding='same')(enc3_3)

        # concatenate
        concate_latent = tf.keras.layers.Concatenate(axis=-1)([latent_1, latent_2, latent_3])

        # add
        # add_latent = tf.keras.layers.add([latent_1, latent_2, latent_3])

        # add convolution blocks (N/16 x N/16 x 16CH)
        # convolution (1x1) to make the same #filters
        shortcut = tf.keras.layers.Conv2D(first_layer_filter_count * 16,
                                          kernel_size=1,
                                          strides=self.CONV_STRIDE,
                                          padding='same')(concate_latent)
        concate_latent = self._add_convolution_block(first_layer_filter_count * 16, concate_latent, 'LeakyReLU')
        concate_latent = self._add_convolution_block(first_layer_filter_count * 16, concate_latent, 'LeakyReLU')
        concate_latent = tf.keras.layers.add([concate_latent, shortcut])

        # -------------------------------------------DECODER PART ------------------------------------------------------
        # (N/8 x N/8 x 8CH)
        dec3 = tf.keras.layers.Dropout(0.5)(concate_latent)
        dec3 = tf.keras.layers.Conv2DTranspose(first_layer_filter_count * 8,
                                               kernel_size=self.DECONV_FILTER_SIZE,
                                               strides=self.DECONV_STRIDE,
                                               kernel_initializer='he_uniform')(dec3)
        dec3 = tf.keras.layers.BatchNormalization()(dec3)
        dec3 = tf.keras.layers.Dropout(0.5)(dec3)
        dec3 = tf.keras.layers.Concatenate(axis=-1)([dec3, enc3_1])

        dec2 = self._add_decoding_block(first_layer_filter_count * 4, True, dec3)       # (N/4 x N/4 x 4CH)
        dec2 = tf.keras.layers.Concatenate(axis=-1)([dec2, enc2_1])

        dec1 = self._add_decoding_block(first_layer_filter_count * 2, True, dec2)       # (N/2 x N/2 x 2CH)
        dec1 = tf.keras.layers.Concatenate(axis=-1)([dec1, enc1_1])

        dec0 = self._add_decoding_block(first_layer_filter_count, True, dec1)           # (N x N x CH)
        dec0 = tf.keras.layers.Concatenate(axis=-1)([dec0, enc0_1])

        # Last layer : CONV BLOCK + convolution + Sigmoid
        last = self._add_convolution_block(first_layer_filter_count, dec0, 'ReLU')
        last = self._add_convolution_block(first_layer_filter_count, last, 'ReLU')
        last = tf.keras.layers.Conv2D(output_channel_count,
                                      kernel_size=self.CONV_FILTER_SIZE,
                                      strides=self.CONV_STRIDE,
                                      padding='same')(last)
        last = tf.keras.layers.Activation(activation='sigmoid')(last)

        self.MulmoUNet = tf.keras.Model(inputs=[input1, input2, input3], outputs=last)

    def _add_encoding_block(self, filter_count, sequence):

        # max pooling
        shortcut = tf.keras.layers.MaxPooling2D(pool_size=self.MAX_POOLING_SIZE,
                                                strides=self.MAX_POOLING_STRIDE,
                                                padding='same')(sequence)

        # add convolution blocks
        new_sequence = self._add_convolution_block(filter_count, shortcut, 'ReLU')
        new_sequence = self._add_convolution_block(filter_count, new_sequence, 'ReLU')

        # convolution (1x1) to make the same #filters
        shortcut = tf.keras.layers.Conv2D(filter_count,
                                          kernel_size=1,
                                          strides=self.CONV_STRIDE,
                                          padding='same')(shortcut)
        new_sequence = tf.keras.layers.add([shortcut, new_sequence])

        return new_sequence

    def _add_decoding_block(self, filter_count, add_drop_layer, sequence):

        new_sequence = self._add_convolution_block(filter_count * 2, sequence, 'ReLU')
        # new_sequence = self._add_convolution_block(filter_count*2, new_sequence, 'ReLU')

        # up-convolution
        new_sequence = tf.keras.layers.Conv2DTranspose(filter_count,
                                                       kernel_size=self.DECONV_FILTER_SIZE,
                                                       strides=self.DECONV_STRIDE,
                                                       kernel_initializer='he_uniform')(new_sequence)
        new_sequence = tf.keras.layers.BatchNormalization()(new_sequence)

        if add_drop_layer:
            new_sequence = tf.keras.layers.Dropout(0.5)(new_sequence)
        return new_sequence

    #  CONV BLOCK : convolution + activation function + batch norm
    def _add_convolution_block(self, filter_count, sequence, activate_function):

        new_sequence = tf.keras.layers.Conv2D(filter_count,
                                              kernel_size=self.CONV_FILTER_SIZE,
                                              strides=self.CONV_STRIDE,
                                              padding='same')(sequence)
        new_sequence = tf.keras.layers.BatchNormalization()(new_sequence)
        if activate_function == 'LeakyReLU':
            new_sequence = tf.keras.layers.LeakyReLU(0.2)(new_sequence)
        elif activate_function == 'ReLU':
            new_sequence = tf.keras.layers.ReLU()(new_sequence)

        return new_sequence

    def get_model(self):
        return self.MulmoUNet
