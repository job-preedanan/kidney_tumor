# https://gist.github.com/RaphaelMeudec/e9a805fa82880876f8d89766f0690b54

import cv2
import numpy as np
import tensorflow as tf

PATH = 'C:/Users/Job/Documents/DoctorProject/Practice_DL/keras-grad-cam/examples'
IMAGE_PATH = '/cat_dog.png'
LAYER_NAME = 'block5_conv3'
CLASS_INDEX = 281


def make_gradcam_heatmap(img_array, modal_select, model, last_conv_layer_name, pred_index=None):
    img_array1 = img_array[0]
    img_array2 = img_array[1]
    img_array3 = img_array[2]

    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model([img_array1])  #, img_array3
        print(preds)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    cam = cv2.resize(heatmap, (224, 224))
    cam = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    # cam = cv2.cvtColor(cam, cv2.COLOR_RGB2BGR)

    if modal_select == 1:
        img = (img_array1[0] + 1) * 127.5
    elif modal_select == 2:
        img = (img_array2[0] + 1) * 127.5
    elif modal_select == 3:
        img = (img_array3[0] + 1) * 127.5
    output_image = cv2.addWeighted(img.astype('uint8'), 0.8, cam, 0.2, 0)
    return output_image


def grad_cam(img, model, layer_name, class_index):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.array([img]))
        loss = predictions[:, class_index]

    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]

    guided_grads = tf.cast(output > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads

    weights = tf.reduce_mean(guided_grads, axis=(0, 1))

    cam = np.ones(output.shape[0: 2], dtype = np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam.numpy(), (224, 224))
    cam = np.maximum(cam, 0)
    heatmap = (cam - cam.min()) / (cam.max() - cam.min())

    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    cam = cv2.cvtColor(cam, cv2.COLOR_RGB2BGR)

    output_image = cv2.addWeighted(cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2BGR), 0.5, cam, 0.5, 0)

    return output_image


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    img = tf.keras.preprocessing.image.load_img(PATH + IMAGE_PATH, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)

    model = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=True)
    cam = make_gradcam_heatmap(img, model, LAYER_NAME, CLASS_INDEX)
    plt.imshow(cam)
    plt.show()