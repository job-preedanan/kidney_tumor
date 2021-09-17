import tensorflow as tf


def make_pretrained_model(height, width, channels, output_channels, pretrained='vgg16'):

    # load pretrained model
    if pretrained == 'resnet':
        base_model1 = tf.keras.applications.resnet.ResNet50(input_shape=(height, width, channels),
                                                            include_top=False,
                                                            weights='imagenet')
    elif pretrained == 'vgg16':
        base_model1 = tf.keras.applications.vgg16.VGG16(input_shape=(height, width, channels),
                                                        include_top=False,
                                                        weights='imagenet')

    for layer in base_model1.layers:
        layer._name = layer.name + str('_m1')

    base_model1.trainable = False
    x1 = base_model1.output

    # x = tf.keras.layers.Concatenate(axis=-1)([x1])

    x = tf.keras.layers.GlobalAveragePooling2D()(x1)
    x = tf.keras.layers.Dense(4096, activation='relu')(x)
    x = tf.keras.layers.Dense(output_channels, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=[base_model1.input], outputs=x)

    return model