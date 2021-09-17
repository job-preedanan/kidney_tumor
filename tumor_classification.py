import numpy as np
import glob
import os
import cv2
import tensorflow as tf
import random
import util_function as utils
import matplotlib.pyplot as plt
from grad_cam import grad_cam, make_gradcam_heatmap
from MulmoNet_vgg_pretrained import make_mulmoNet_vgg16
from pretrained_model import make_pretrained_model
from data_loader import load_data_from_xlsx


PATH = 'C:/Users/Job/Documents/DoctorProject/kidney_tumor'
DATA_FOLDER = '/dataset/'
EXPORT_FOLDER = '/exports/classification/cropped_tumor/pretrained_vgg16_stack/3_modals/'
IMAGE_SIZE = 224
NUM_CHANNELS = 3
BATCH_SIZE = 16
EPOCH = 100
CROP_RATIO = 0


# histogram equalization + cropping + resize + normalize
def preprocessing(image, histeq=True):

    # hist equalization
    if histeq:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)

    crop_size = round(image.shape[1]*CROP_RATIO)
    image = image[:, crop_size:image.shape[1] - crop_size]    # cropping
    image = np.array(utils.normalize_x(cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))), np.float32)  # normalize(-1, 1)

    return image


# data spliting function
def split_train_test(x, y, val_ratio=0.2, random_sample=True):

    # zip x and y
    samples = list(zip(x, y))
    if random_sample:
        random.shuffle(samples)

    split_idx = int(round(len(samples) * val_ratio))  # split index
    test = samples[:split_idx]
    train = samples[split_idx:]

    # unzip and convert to array
    x_train, y_train = zip(*train)
    x_test, y_test = zip(*test)
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)

    return x_train, y_train, x_test, y_test


def train(x_train, y_train, x_val, y_val, modal_name, mulmonet=True):

    modal_name = modal_name[0] + '_' + modal_name[1] + '_' + modal_name[2]

    try:
        os.makedirs(PATH + EXPORT_FOLDER + modal_name)
    except(FileExistsError):
        print('folders exist')

    if mulmonet:
        # split modals
        x_train1 = x_train[:, :, :, :3]
        x_train2 = x_train[:, :, :, 3:6]
        x_train3 = x_train[:, :, :, 6:]
        x_val1 = x_val[:, :, :, :3]
        x_val2 = x_val[:, :, :, 3:6]
        x_val3 = x_val[:, :, :, 6:]

        model = make_mulmoNet_vgg16(IMAGE_SIZE, IMAGE_SIZE, 3, 1)
        model.summary()

    else:
        # stack modals
        x_train1 = np.zeros_like(x_train[:, :, :, :3])
        x_train1[:, :, :, 0] = x_train[:, :, :, 0]
        x_train1[:, :, :, 1] = x_train[:, :, :, 3]
        x_train1[:, :, :, 2] = x_train[:, :, :, 6]

        x_val1 = np.zeros_like(x_val[:, :, :, :3])
        x_val1[:, :, :, 0] = x_val[:, :, :, 0]
        x_val1[:, :, :, 1] = x_val[:, :, :, 3]
        x_val1[:, :, :, 2] = x_val[:, :, :, 6]

        model = make_pretrained_model(IMAGE_SIZE, IMAGE_SIZE, 3, 1)
        model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # data augmentation
    aug = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=5,
                                                          fill_mode='nearest',
                                                          vertical_flip=True,
                                                          horizontal_flip=True)
    # learning rate decay callback
    lr_reduce_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                              mode='min',
                                                              factor=0.5,
                                                              patience=10,
                                                              min_lr=5e-5)

    # model checkpoint
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=PATH + EXPORT_FOLDER + modal_name + '/weight_checkpoint.hdf5',
                                                                   save_weights_only=True,
                                                                   monitor='loss',
                                                                   mode='min',
                                                                   save_best_only=True)

    # generator flow (3 modal inputs)
    def generator(x1, x2, x3, y, batch_size):
        gen_1 = aug.flow(x1, y, batch_size=batch_size, seed=1)
        gen_2 = aug.flow(x2, y, batch_size=batch_size, seed=1)
        gen_3 = aug.flow(x3, y, batch_size=batch_size, seed=1)
        while True:
            x1 = gen_1.next()
            x2 = gen_2.next()
            x3 = gen_3.next()

            # ((x1, x2, x3), y1)
            yield [x1[0], x2[0], x3[0]], x1[1]

    # fits the model on batches with real-time data augmentation:
    history = model.fit(generator(x_train1, x_train2, x_train3, y_train, batch_size=BATCH_SIZE),
                        validation_data=([x_val1, x_val2, x_val3], y_val),
                        steps_per_epoch=len(x_train) // BATCH_SIZE,
                        callbacks=[lr_reduce_callback, model_checkpoint_callback],
                        epochs=EPOCH,
                        verbose=1)

    model.save_weights(PATH + EXPORT_FOLDER + modal_name + '/model_weights.hdf5')

    #  "Accuracy"
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(PATH + EXPORT_FOLDER + modal_name + '/multimodal_model_accuracy.png')
    plt.clf()

    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(PATH + EXPORT_FOLDER + modal_name + '/multimodal_model_loss.png')
    plt.clf()

    test_loss, test_acc = model.evaluate([x_val1, x_val2, x_val3], y_val)
    print('Test accuracy:', test_acc)


def predict(x_test, y_test, modal_name):
    from sklearn.metrics import confusion_matrix, roc_curve, auc, RocCurveDisplay
    import pandas as pd
    import seaborn as sn

    modal_name = modal_name[0] + '_' + modal_name[1] + '_' + modal_name[2]

    if mulmonet:
        # split modals
        x_test1 = x_test[:, :, :, :3]
        x_test2 = x_test[:, :, :, 3:6]
        x_test3 = x_test[:, :, :, 6:9]
    else:
        # stack
        x_test1 = np.zeros_like(x_test[:, :, :, :3])
        x_test1[:, :, :, 0] = x_test[:, :, :, 0]
        x_test1[:, :, :, 1] = x_test[:, :, :, 3]
        x_test1[:, :, :, 2] = x_test[:, :, :, 6]



    # load model and weights
    model = make_mulmoNet_vgg16(IMAGE_SIZE, IMAGE_SIZE, 3, 1)
    model.summary()
    model.load_weights(PATH + EXPORT_FOLDER + modal_name + '/weight_checkpoint.hdf5')

    # prediction
    y_pred = model.predict([x_test1, x_test2, x_test3], BATCH_SIZE)
    y_pred_binary = np.rint(y_pred).astype(int)

    # confusion matrix
    confusion_matrix = confusion_matrix(y_test, y_pred_binary)
    print(confusion_matrix)

    df_cm = pd.DataFrame(confusion_matrix, range(2), range(2))
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size
    plt.savefig(PATH + EXPORT_FOLDER + modal_name + '/confusion_matrix.png')
    plt.clf()

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=modal_name)
    display.plot()
    plt.savefig(PATH + EXPORT_FOLDER + modal_name + '/ruc_curve.png')

    y_test = np.concatenate(y_test)
    with open(PATH + EXPORT_FOLDER + modal_name + '/results.txt', 'w') as f:
        f.write('TP: ' + str(confusion_matrix[1, 1]))
        f.write('\n')
        f.write('FP: ' + str(confusion_matrix[0, 1]))
        f.write('\n')
        f.write('FN: ' + str(confusion_matrix[1, 0]))
        f.write('\n')
        f.write('TN: ' + str(confusion_matrix[0, 0]))

    try:
        os.makedirs(PATH + EXPORT_FOLDER + modal_name + '/results')
    except(FileExistsError):
        print('folders exist')

    # GRAD CAM
    for i, y in enumerate(y_pred):
        img1 = x_test1[i]
        img2 = x_test1[i, :, :, 1]
        img3 = x_test1[i, :, :, 2]

        image_array1 = np.expand_dims(img1, axis=0)
        image_array2 = np.expand_dims(img2, axis=0)
        image_array3 = np.expand_dims(img3, axis=0)
        cam1 = make_gradcam_heatmap([image_array1, image_array2, image_array3], 1, model, 'block5_conv3_m1', 0)
        cam2 = make_gradcam_heatmap([image_array1, image_array2, image_array3], 2, model, 'block5_conv3_m2', 0)
        cam3 = make_gradcam_heatmap([image_array1, image_array2, image_array3], 3, model, 'block5_conv3_m3', 0)

        # f, axarr = plt.subplots(3, 2)
        # axarr[0, 0].imshow(utils.denormalize_x(img1), cmap='gray', vmin=0, vmax=255)
        # axarr[0, 1].imshow(cam1)
        # axarr[1, 0].imshow(utils.denormalize_x(img2), cmap='gray', vmin=0, vmax=255)
        # axarr[1, 1].imshow(cam2)
        # axarr[2, 0].imshow(utils.denormalize_x(img3), cmap='gray', vmin=0, vmax=255)
        # axarr[2, 1].imshow(cam3)

        # plt.title(test_names[i])
        # plt.show()

        correct = 1 if y_pred_binary[i] == y_test[i] else 0

        img1 = (img1 + 1) * 127.5
        img2 = (img2 + 1) * 127.5
        img3 = (img3 + 1) * 127.5
        # save_image = np.concatenate((np.repeat(np.expand_dims(img1[:, :, 0], axis=-1), 3, axis=-1),
        #                             np.repeat(np.expand_dims(img1[:, :, 1], axis=-1), 3, axis=-1),
        #                             np.repeat(np.expand_dims(img1[:, :, 2], axis=-1), 3, axis=-1),
        #                             cam1), axis=1)
        save_image1 = np.concatenate((img1, cam1), axis=1)
        save_image2 = np.concatenate((img2, cam2), axis=1)
        save_image3 = np.concatenate((img3, cam3), axis=1)
        save_image = np.concatenate((save_image1, save_image2), axis=0)
        save_image = np.concatenate((save_image, save_image3), axis=0)

        # save_image = np.concatenate((cam1, cam2, cam3), axis=1)

        cv2.imwrite(PATH + EXPORT_FOLDER + modal_name + '/results/' + str(i) + '_' + str(correct) + str(np.round(y_pred[i], 2)) + '.png', save_image)


if __name__ == '__main__':
    print(tf.__version__)
    import matplotlib.pyplot as plt

    # modal_lists = [['pc', 'dc'], ['pc', 'tm'], ['pc', 'am'],
    #                ['ec', 'dc'], ['ec', 'tm'], ['ec', 'am'],
    #                ['dc', 'tm'], ['dc', 'am'],
    #                ['tm', 'am']]

    modal_lists = [['pc', 'ec', 'dc'], ['pc', 'ec', 'tm'], ['pc', 'ec', 'am'],
                   ['pc', 'dc', 'tm'], ['pc', 'dc', 'am'], ['pc', 'tm', 'am'],
                   ['ec', 'dc', 'tm'], ['ec', 'dc', 'am'], ['ec', 'tm', 'am'],
                   ['dc', 'tm', 'am']]

    for modal_name in modal_lists:
        print(modal_name)
        x_data, labels = load_data_from_xlsx(PATH + DATA_FOLDER, modal_lists=modal_name, image_type=1)
        y_data = labels
        # y_data, _ = load_data_from_xlsx(PATH + DATA_FOLDER, modal_lists=modal_name, image_type=0)
        print(x_data.shape)
        print(labels.shape)

        # # display sample images
        # f, axarr = plt.subplots(2, len(modal_lists))
        # idx = random.randint(0, len(x_data))
        # for m, modal in enumerate(modal_lists):
        #     axarr[0, m].set_title(modal)
        #     axarr[0, m].grid(False)
        #     axarr[0, m].imshow(utils.denormalize_x(x_data[idx, :, :, m*3:m*3 + 3]), cmap='gray', vmin=0, vmax=255)
        #     axarr[1, m].imshow(utils.denormalize_y(y_data[idx, :, :, m]), cmap='gray', vmin=0, vmax=255)
        #
        # plt.show()

        # split train - test
        x_train_val, y_train_val, x_test, y_test = split_train_test(x_data, y_data, val_ratio=0.2, random_sample=False)
        x_train, y_train, x_val, y_val = split_train_test(x_train_val, y_train_val, val_ratio=0.125, random_sample=False)
        print(x_train.shape)
        print(y_train.shape)
        print(x_val.shape)
        print(y_val.shape)
        print(x_test.shape)
        print(y_test.shape)

        train(x_train, y_train, x_val, y_val, modal_name)
        predict(x_test, y_test, modal_name)

