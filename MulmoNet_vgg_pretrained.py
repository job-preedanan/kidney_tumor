import tensorflow as tf


def make_mulmoNet_vgg19(height, width, channels, output_channels):

    # load pretrained model
    base_model1 = tf.keras.applications.vgg19.VGG19(input_shape=(height, width, channels),
                                                    include_top=False,
                                                    weights='imagenet')
    for layer in base_model1.layers:
        layer._name = layer.name + str('_m1')
    # Depth Model

    base_model2 = tf.keras.applications.vgg19.VGG19(input_shape=(height, width, channels),
                                                    include_top=False,
                                                    weights='imagenet')

    for layer in base_model2.layers:
        layer._name = layer.name + str('_m2')

    base_model1.trainable = False
    base_model2.trainable = False
    x1 = base_model1.output
    x2 = base_model2.output

    x = tf.keras.layers.Concatenate(axis=-1)([x1, x2])

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(4096, activation='relu')(x)
    x = tf.keras.layers.Dense(output_channels, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=[base_model1.input, base_model2.input], outputs=x)

    return model


def make_mulmoNet_vgg16(height, width, channels, output_channels):

    # load pretrained model
    base_model1 = tf.keras.applications.vgg16.VGG16(input_shape=(height, width, channels),
                                                    include_top=False,
                                                    weights='imagenet')
    for layer in base_model1.layers:
        layer._name = layer.name + str('_m1')

    base_model2 = tf.keras.applications.vgg16.VGG16(input_shape=(height, width, channels),
                                                    include_top=False,
                                                    weights='imagenet')

    for layer in base_model2.layers:
        layer._name = layer.name + str('_m2')

    base_model3 = tf.keras.applications.vgg16.VGG16(input_shape=(height, width, channels),
                                                    include_top=False,
                                                    weights='imagenet')

    for layer in base_model3.layers:
        layer._name = layer.name + str('_m3')

    base_model1.trainable = False
    base_model2.trainable = False
    base_model3.trainable = False
    latent_1 = base_model1.get_layer('block5_conv3_m1').output
    latent_2 = base_model2.get_layer('block5_conv3_m2').output
    latent_3 = base_model3.get_layer('block5_conv3_m3').output

    concate_latent = tf.keras.layers.Concatenate(axis=-1)([latent_1, latent_2, latent_3])

    dense = tf.keras.layers.GlobalAveragePooling2D()(latent_1)
    dense = tf.keras.layers.Dense(4096, activation='relu')(dense)
    dense = tf.keras.layers.Dense(4096, activation='relu')(dense)
    output = tf.keras.layers.Dense(output_channels, activation='sigmoid')(dense)

    model = tf.keras.Model(inputs=[base_model1.input], outputs=output)

    return model


def make_mulmoUNet_vgg16(height, width, channels, output_channels):

    def decoding_block(filter_count, sequence):

        new_sequence = convolution_block(filter_count * 2, sequence)
        new_sequence = convolution_block(filter_count * 2, new_sequence)

        # up-convolution
        new_sequence = tf.keras.layers.Conv2DTranspose(filter_count,
                                                       kernel_size=2,
                                                       strides=2,
                                                       kernel_initializer='he_uniform')(new_sequence)
        new_sequence = tf.keras.layers.BatchNormalization()(new_sequence)

        return new_sequence

    #  CONV BLOCK : convolution + activation function + batch norm
    def convolution_block(filter_count, sequence):

        new_sequence = tf.keras.layers.Conv2D(filter_count, kernel_size=3, strides=1, padding='same')(sequence)
        new_sequence = tf.keras.layers.BatchNormalization()(new_sequence)
        new_sequence = tf.keras.layers.ReLU()(new_sequence)

        return new_sequence

    # load pretrained model 1
    base_model1 = tf.keras.applications.vgg16.VGG16(input_shape=(height, width, channels),
                                                    include_top=False,
                                                    weights='imagenet')
    for layer in base_model1.layers:
        layer._name = layer.name + str('_m1')

    # load pretrained model 2
    base_model2 = tf.keras.applications.vgg16.VGG16(input_shape=(height, width, channels),
                                                    include_top=False,
                                                    weights='imagenet')
    for layer in base_model2.layers:
        layer._name = layer.name + str('_m2')

    # load pretrained model 3
    base_model3 = tf.keras.applications.vgg16.VGG16(input_shape=(height, width, channels),
                                                    include_top=False,
                                                    weights='imagenet')

    for layer in base_model3.layers:
        layer._name = layer.name + str('_m3')

    base_model1.trainable = False
    base_model2.trainable = False
    base_model3.trainable = False

    lat_output1 = base_model1.get_layer('block5_conv3_m1').output
    lat_output2 = base_model2.get_layer('block5_conv3_m2').output
    lat_output3 = base_model3.get_layer('block5_conv3_m3').output
    vgg_output_concate = tf.keras.layers.Concatenate(axis=-1)([lat_output1, lat_output2, lat_output3])

    # -------------------------------------------DECODER PART ------------------------------------------------------

    enc0 = base_model1.get_layer('block1_conv2_m1').output
    enc1 = base_model1.get_layer('block2_conv2_m1').output
    enc2 = base_model1.get_layer('block3_conv2_m1').output
    enc3 = base_model1.get_layer('block4_conv3_m1').output

    first_layer_filter_count = 64

    # (N/8 x N/8 x 8CH)
    dec3 = tf.keras.layers.Conv2DTranspose(first_layer_filter_count * 8,
                                           kernel_size=2,
                                           strides=2,
                                           kernel_initializer='he_uniform')(vgg_output_concate)
    dec3 = tf.keras.layers.BatchNormalization()(dec3)
    dec3 = tf.keras.layers.Concatenate(axis=-1)([dec3, enc3])

    dec2 = decoding_block(first_layer_filter_count * 4, dec3)  # (N/4 x N/4 x 4CH)
    dec2 = tf.keras.layers.Concatenate(axis=-1)([dec2, enc2])

    dec1 = decoding_block(first_layer_filter_count * 2, dec2)  # (N/2 x N/2 x 2CH)
    dec1 = tf.keras.layers.Concatenate(axis=-1)([dec1, enc1])

    dec0 = decoding_block(first_layer_filter_count, dec1)  # (N x N x CH)
    dec0 = tf.keras.layers.Concatenate(axis=-1)([dec0, enc0])

    # Last layer : CONV BLOCK + convolution + Sigmoid
    last = convolution_block(first_layer_filter_count, dec0)
    last = convolution_block(first_layer_filter_count, last)
    last = tf.keras.layers.Conv2D(output_channels, kernel_size=3, strides=1, padding='same')(last)
    last = tf.keras.layers.Activation(activation='sigmoid')(last)

    model = tf.keras.Model(inputs=[base_model1.input, base_model2.input, base_model3.input], outputs=last)

    return model


def make_mulmoXNet_vgg16(height, width, channels, output_channels):

    def decoding_block(filter_count, sequence):

        new_sequence = convolution_block(filter_count * 2, sequence)
        new_sequence = convolution_block(filter_count * 2, new_sequence)

        # up-convolution
        new_sequence = tf.keras.layers.Conv2DTranspose(filter_count,
                                                       kernel_size=2,
                                                       strides=2,
                                                       kernel_initializer='he_uniform')(new_sequence)
        new_sequence = tf.keras.layers.BatchNormalization()(new_sequence)

        return new_sequence

    #  CONV BLOCK : convolution + activation function + batch norm
    def convolution_block(filter_count, sequence):

        new_sequence = tf.keras.layers.Conv2D(filter_count, kernel_size=3, strides=1, padding='same')(sequence)
        new_sequence = tf.keras.layers.BatchNormalization()(new_sequence)
        new_sequence = tf.keras.layers.ReLU()(new_sequence)

        return new_sequence

    # load pretrained model
    base_model1 = tf.keras.applications.vgg16.VGG16(input_shape=(height, width, channels),
                                                    include_top=False,
                                                    weights='imagenet')
    for layer in base_model1.layers:
        layer._name = layer.name + str('_m1')
    # Depth Model

    base_model2 = tf.keras.applications.vgg16.VGG16(input_shape=(height, width, channels),
                                                    include_top=False,
                                                    weights='imagenet')

    for layer in base_model2.layers:
        layer._name = layer.name + str('_m2')

    base_model1.trainable = False
    base_model2.trainable = False

    lat_output1 = base_model1.get_layer('block5_conv3_m1').output
    lat_output2 = base_model2.get_layer('block5_conv3_m2').output
    vgg_output_concate = tf.keras.layers.Concatenate(axis=-1)([lat_output1, lat_output2])

    # -------------------------------------------DECODER 1 ------------------------------------------------------

    enc0_m1 = base_model1.get_layer('block1_conv2_m1').output
    enc1_m1 = base_model1.get_layer('block2_conv2_m1').output
    enc2_m1 = base_model1.get_layer('block3_conv2_m1').output
    enc3_m1 = base_model1.get_layer('block4_conv3_m1').output

    first_layer_filter_count = 64

    # (N/8 x N/8 x 8CH)
    dec3_m1 = tf.keras.layers.Conv2DTranspose(first_layer_filter_count * 8,
                                              kernel_size=2,
                                              strides=2,
                                              kernel_initializer='he_uniform')(vgg_output_concate)
    dec3_m1 = tf.keras.layers.BatchNormalization()(dec3_m1)
    dec3_m1 = tf.keras.layers.Concatenate(axis=-1)([dec3_m1, enc3_m1])

    dec2_m1 = decoding_block(first_layer_filter_count * 4, dec3_m1)  # (N/4 x N/4 x 4CH)
    dec2_m1 = tf.keras.layers.Concatenate(axis=-1)([dec2_m1, enc2_m1])

    dec1_m1 = decoding_block(first_layer_filter_count * 2, dec2_m1)  # (N/2 x N/2 x 2CH)
    dec1_m1 = tf.keras.layers.Concatenate(axis=-1)([dec1_m1, enc1_m1])

    dec0_m1 = decoding_block(first_layer_filter_count, dec1_m1)  # (N x N x CH)
    dec0_m1 = tf.keras.layers.Concatenate(axis=-1)([dec0_m1, enc0_m1])

    # Last layer : CONV BLOCK + convolution + Sigmoid
    output1 = convolution_block(first_layer_filter_count, dec0_m1)
    output1 = convolution_block(first_layer_filter_count, output1)
    output1 = tf.keras.layers.Conv2D(output_channels, kernel_size=3, strides=1, padding='same')(output1)
    output1 = tf.keras.layers.Activation(activation='sigmoid')(output1)

    # -------------------------------------------DECODER 1 ------------------------------------------------------

    enc0_m2 = base_model2.get_layer('block1_conv2_m2').output
    enc1_m2 = base_model2.get_layer('block2_conv2_m2').output
    enc2_m2 = base_model2.get_layer('block3_conv2_m2').output
    enc3_m2 = base_model2.get_layer('block4_conv3_m2').output

    first_layer_filter_count = 64

    # (N/8 x N/8 x 8CH)
    dec3_m2 = tf.keras.layers.Conv2DTranspose(first_layer_filter_count * 8,
                                              kernel_size=2,
                                              strides=2,
                                              kernel_initializer='he_uniform')(vgg_output_concate)
    dec3_m2 = tf.keras.layers.BatchNormalization()(dec3_m2)
    dec3_m2 = tf.keras.layers.Concatenate(axis=-1)([dec3_m2, enc3_m2])

    dec2_m2 = decoding_block(first_layer_filter_count * 4, dec3_m2)  # (N/4 x N/4 x 4CH)
    dec2_m2 = tf.keras.layers.Concatenate(axis=-1)([dec2_m2, enc2_m2])

    dec1_m2 = decoding_block(first_layer_filter_count * 2, dec2_m2)  # (N/2 x N/2 x 2CH)
    dec1_m2 = tf.keras.layers.Concatenate(axis=-1)([dec1_m2, enc1_m2])

    dec0_m2 = decoding_block(first_layer_filter_count, dec1_m2)  # (N x N x CH)
    dec0_m2 = tf.keras.layers.Concatenate(axis=-1)([dec0_m2, enc0_m2])

    # Last layer : CONV BLOCK + convolution + Sigmoid
    output2 = convolution_block(first_layer_filter_count, dec0_m2)
    output2 = convolution_block(first_layer_filter_count, output2)
    output2 = tf.keras.layers.Conv2D(output_channels, kernel_size=3, strides=1, padding='same')(output2)
    output2 = tf.keras.layers.Activation(activation='sigmoid')(output2)

    model = tf.keras.Model(inputs=[base_model1.input, base_model2.input], outputs=[output1, output2])

    return model


if __name__ == '__main__':
    model = make_mulmoUNet_vgg16(224, 224, 3, 1)
    model.summary()