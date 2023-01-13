import numpy as np
import glob
import os
import cv2
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import random
import util_function as utils
import pandas as pd
import matplotlib.pyplot as plt
from grad_cam import grad_cam, make_gradcam_heatmap
# from MulmoNet_vgg_pretrained import make_mulmoNet_vgg16
from pretrained_model import make_pretrained_model
from data_loader import load_data_from_xlsx


PATH = 'C:/Users/Job/Documents/DoctorProject/kidney_tumor'
DATA_FOLDER = '/dataset/'
EXPORT_FOLDER = '/exports/classification/cropped_tumor/dataset2023/pretrained_vgg/2_modal/'               #_
IMAGE_SIZE = 224
NUM_CHANNELS = 3
BATCH_SIZE = 16
EPOCH = 100
CROP_RATIO = 0
CROSS_NUM = 0
NUM_CLASSES = 2


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


# list moving function
def list_index_move(list, split_num):
    split_idx = int(round(len(list) * split_num))  # split index
    new_list = list[split_idx:]
    new_list = np.concatenate([new_list, list[:split_idx]])
    return new_list


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


def class_weight(labels):
    # class rebalance : normal vs tumor
    image_class = np.unique(labels)
    class_weight = {}

    for i in image_class:
        class_weight[i] = len(labels) / np.sum(labels == i)

    return class_weight


def train(x_train, y_train, x_val, y_val, modal_name, mulmonet=False):

    modal_name = modal_name[0] + '_' + modal_name[1] #+ '_' + modal_name[2]

    try:
        os.makedirs(PATH + EXPORT_FOLDER + modal_name + '/cross_val#' + str(CROSS_NUM))
    except(FileExistsError):
        print('folders exist')

    if mulmonet:
        # ct models
        # x_train1 = np.zeros_like(x_train[:, :, :, :3])
        # x_train1[:, :, :, 0] = x_train[:, :, :, 0]
        # x_train1[:, :, :, 1] = x_train[:, :, :, 3]
        # x_train1[:, :, :, 2] = x_train[:, :, :, 6]
        #
        # # mri modals
        # x_train2 = np.zeros_like(x_train[:, :, :, :3])
        # x_train2[:, :, :, 0] = x_train[:, :, :, 9]
        # x_train2[:, :, :, 1] = x_train[:, :, :, 12]

        # # ct models - val
        # x_val1 = np.zeros_like(x_val[:, :, :, :3])
        # x_val1[:, :, :, 0] = x_val[:, :, :, 0]
        # x_val1[:, :, :, 1] = x_val[:, :, :, 3]
        # x_val1[:, :, :, 2] = x_val[:, :, :, 6]
        #
        # # mri modals - val
        # x_val2 = np.zeros_like(x_val[:, :, :, :3])
        # x_val2[:, :, :, 0] = x_val[:, :, :, 9]
        # x_val2[:, :, :, 1] = x_val[:, :, :, 12]

        x_train1 = x_train[:, :, :, :3]
        x_train2 = x_train[:, :, :, 3:6]
        x_train3 = x_train[:, :, :, 6:9]
        # x_train4 = x_train[:, :, :, 9:12]
        # x_train5 = x_train[:, :, :, 12:15]

        x_val1 = x_val[:, :, :, :3]
        x_val2 = x_val[:, :, :, 3:6]
        x_val3 = x_val[:, :, :, 6:9]
        # x_val4 = x_val[:, :, :, 9:12]
        # x_val5 = x_val[:, :, :, 12:15]

        # model = make_mulmoNet_vgg16(2, IMAGE_SIZE, IMAGE_SIZE, 3, 1)
        # model.summary()

    else:
        # stack modals
        x_train1 = np.zeros_like(x_train[:, :, :, :3])
        x_train1[:, :, :, 0] = x_train[:, :, :, 0]
        x_train1[:, :, :, 1] = x_train[:, :, :, 3]
        # x_train1[:, :, :, 2] = x_train[:, :, :, 6]

        x_val1 = np.zeros_like(x_val[:, :, :, :3])
        x_val1[:, :, :, 0] = x_val[:, :, :, 0]
        x_val1[:, :, :, 1] = x_val[:, :, :, 3]
        # x_val1[:, :, :, 2] = x_val[:, :, :, 6]

        model = make_pretrained_model(IMAGE_SIZE, IMAGE_SIZE, 3, 1)
        model.summary()

    l_weight = class_weight(y_train)
    print(l_weight)

    # train_label = to_categorical(y_train, num_classes=NUM_CLASSES)
    # val_label = to_categorical(y_val, num_classes=NUM_CLASSES)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',  #tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    # data augmentation
    aug = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=10,
                                                          fill_mode='nearest',
                                                          vertical_flip=True,
                                                          horizontal_flip=True)
    # learning rate decay callback
    lr_reduce_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                              mode='min',
                                                              factor=0.5,
                                                              patience=10,
                                                              min_lr=5e-5)

    # model checkpoint
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=PATH + EXPORT_FOLDER + modal_name
                                                                            + '/cross_val#' + str(CROSS_NUM)
                                                                            + '/weight_checkpoint.hdf5',
                                                                   save_weights_only=True,
                                                                   monitor='val_loss',
                                                                   mode='min',
                                                                   save_best_only=True)

    if mulmonet:
        # generator flow (3 modal inputs)
        def generator(x1, x2, y, batch_size):
            gen_1 = aug.flow(x1, y, batch_size=batch_size, seed=1)
            gen_2 = aug.flow(x2, y, batch_size=batch_size, seed=1)
            # gen_3 = aug.flow(x3, y, batch_size=batch_size, seed=1)
            # gen_4 = aug.flow(x4, y, batch_size=batch_size, seed=1)
            # gen_5 = aug.flow(x5, y, batch_size=batch_size, seed=1)
            while True:
                x1 = gen_1.next()
                x2 = gen_2.next()
                # x3 = gen_3.next()
                # x4 = gen_4.next()
                # x5 = gen_5.next()

                # ((x1, x2), y1)
                yield [x1[0], x2[0]], x1[1]

        # fits the model on batches with real-time data augmentation:
        history = model.fit(generator(x_train1,
                                      x_train2,
                                      y_train,
                                      batch_size=BATCH_SIZE),
                            validation_data=([x_val1, x_val2], y_val),
                            steps_per_epoch=len(x_train) // BATCH_SIZE,
                            callbacks=[lr_reduce_callback, model_checkpoint_callback],
                            class_weight=l_weight,
                            epochs=EPOCH,
                            verbose=1)
    else:
        # generator flow (3 modal inputs)
        def generator(x1, y, batch_size):
            gen_1 = aug.flow(x1, y, batch_size=batch_size, seed=1)

            while True:
                x1 = gen_1.next()

                yield x1[0], x1[1]

        # fits the model on batches with real-time data augmentation:
        history = model.fit(generator(x_train1, y_train, batch_size=BATCH_SIZE),
                            validation_data=(x_val1, y_val),
                            steps_per_epoch=len(x_train) // BATCH_SIZE,
                            callbacks=[lr_reduce_callback, model_checkpoint_callback],
                            class_weight=l_weight,
                            epochs=EPOCH,
                            verbose=1)

    model.save_weights(PATH + EXPORT_FOLDER + modal_name + '/cross_val#' + str(CROSS_NUM) + '/model_weights.hdf5')

    #  "Accuracy"
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(PATH + EXPORT_FOLDER + modal_name + '/cross_val#' + str(CROSS_NUM) + '/model_accuracy.png')
    plt.clf()

    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(PATH + EXPORT_FOLDER + modal_name + '/cross_val#' + str(CROSS_NUM) + '/model_loss.png')
    plt.clf()

    # test_loss, test_acc = model.evaluate([x_val1], y_val)
    # print('Test accuracy:', test_acc)


def predict(x_test, y_test, modal_name, mulmonet=False):
    from sklearn.metrics import confusion_matrix, roc_curve, auc, RocCurveDisplay
    import pandas as pd
    import seaborn as sn
    # from skimage.segmentation import mark_boundaries
    # from lime import lime_image

    modal_name = modal_name[0] + '_' + modal_name[1] #+ '_' + modal_name[2]

    if mulmonet:
        # # ct models
        # x_test1 = np.zeros_like(x_test[:, :, :, :3])
        # x_test1[:, :, :, 0] = x_test[:, :, :, 0]
        # x_test1[:, :, :, 1] = x_test[:, :, :, 3]
        # x_test1[:, :, :, 2] = x_test[:, :, :, 6]
        #
        # # mri modals
        # x_test2 = np.zeros_like(x_test[:, :, :, :3])
        # x_test2[:, :, :, 0] = x_test[:, :, :, 9]
        # x_test2[:, :, :, 1] = x_test[:, :, :, 12]

        x_test1 = x_test[:, :, :, :3]
        # x_test2 = x_test[:, :, :, 3:6]
        # x_test3 = x_test[:, :, :, 6:9]
        # x_test4 = x_test[:, :, :, 9:12]
        # x_test5 = x_test[:, :, :, 12:15]

        # load model and weights
        # model = make_mulmoNet_vgg16(2, IMAGE_SIZE, IMAGE_SIZE, 3, 1)
        # model.summary()
        # model.load_weights(PATH + EXPORT_FOLDER + modal_name + '/cross_val#' + str(CROSS_NUM) + '/weight_checkpoint.hdf5')

        # prediction
        # y_pred = model.predict([x_test1, x_test2], BATCH_SIZE)
        # y_pred_binary = np.rint(y_pred).astype(int)
    else:
        # stack
        # x_test1 = x_test
        x_test1 = np.zeros_like(x_test[:, :, :, :3])
        x_test1[:, :, :, 0] = x_test[:, :, :, 0]
        x_test1[:, :, :, 1] = x_test[:, :, :, 3]
        # x_test1[:, :, :, 2] = x_test[:, :, :, 6]

        # load model and weights
        model = make_pretrained_model(IMAGE_SIZE, IMAGE_SIZE, 3, 1)
        model.summary()
        model.load_weights(PATH + EXPORT_FOLDER + modal_name + '/cross_val#' + str(CROSS_NUM) + '/weight_checkpoint.hdf5')

        # prediction
        y_pred_logit = model.predict(x_test1, BATCH_SIZE)

    if NUM_CLASSES == 5:
        y_test = to_categorical(y_test)

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(NUM_CLASSES):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_logit[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot of a ROC curve for a specific class
        for i in range(NUM_CLASSES):
            plt.figure()
            plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('FPR')
            plt.ylabel('TPR')
            plt.title('ROC of class#' + str(i))
            plt.legend(loc="lower right")
            # plt.savefig(PATH + EXPORT_FOLDER + modal_name + '/cross_val#' + str(CROSS_NUM) + '/ruc_curve_' + str(i) + '.png')
            # plt.clf()

        y_pred = np.zeros(len(y_pred_logit))
        for i in range(len(y_pred_logit)):
            y_pred[i] = np.argmax(y_pred_logit[i])

    else:
        # AUC
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_logit)
        best_point = np.argmax((1-fpr)*tpr)
        best_th = thresholds[best_point]
        roc_auc = auc(fpr, tpr)
        display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=modal_name)
        display.plot()
        plt.savefig(PATH + EXPORT_FOLDER + modal_name + '/cross_val#' + str(CROSS_NUM) + '/ruc_curve.png')
        plt.clf()

        # thresholding
        y_pred = y_pred_logit.copy()
        y_pred[y_pred >= best_th] = 1
        y_pred[y_pred < best_th] = 0

    # confusion matrix
    confusion_matrix = confusion_matrix(y_test, y_pred)
    print(confusion_matrix)

    df_cm = pd.DataFrame(confusion_matrix, range(2), range(2))
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size
    plt.savefig(PATH + EXPORT_FOLDER + modal_name + '/cross_val#' + str(CROSS_NUM) + '/confusion_matrix.png')
    plt.clf()

    with open(PATH + EXPORT_FOLDER + modal_name + '/cross_val#' + str(CROSS_NUM) + '/results.txt', 'w') as f:
        f.write('TP: ' + str(confusion_matrix[1, 1]))
        f.write('\n')
        f.write('FP: ' + str(confusion_matrix[0, 1]))
        f.write('\n')
        f.write('FN: ' + str(confusion_matrix[1, 0]))
        f.write('\n')
        f.write('TN: ' + str(confusion_matrix[0, 0]))

    print(np.sum(confusion_matrix))
    sensitivity = round(confusion_matrix[1, 1]/(confusion_matrix[1, 1] + confusion_matrix[1, 0]), 2)
    specificity = round(confusion_matrix[0, 0]/(confusion_matrix[0, 0] + confusion_matrix[0, 1]), 2)
    acc = round((confusion_matrix[1, 1] + confusion_matrix[0, 0]) / np.sum(confusion_matrix), 2)

    # try:
    #     os.makedirs(PATH + EXPORT_FOLDER + modal_name + '/cross_val#' + str(CROSS_NUM) + '/results')
    # except(FileExistsError):
    #     print('folders exist')

    # # LIME
    #
    # try:
    #     os.makedirs(PATH + EXPORT_FOLDER + modal_name + '/lime_hm')
    #     os.makedirs(PATH + EXPORT_FOLDER + modal_name + '/lime')
    # except(FileExistsError):
    #     print('folders exist')
    #
    # explainer = lime_image.LimeImageExplainer()
    # for i in range(len(x_test)):
    #     correct = 1 if y_pred_binary[i] == y_test[i] else 0
    #
    #     explanation = explainer.explain_instance(x_test[i].astype('double'),
    #                                              model.predict,
    #                                              top_labels=2,
    #                                              hide_color=0,
    #                                              num_samples=1000)
    #
    #     temp, mask = explanation.get_image_and_mask(explanation.top_labels[0],
    #                                                 positive_only=False,
    #                                                 num_features=10,
    #                                                 hide_rest=False)
    #
    #     lime_image = mark_boundaries(temp / 2 + 0.5, mask)
    #     # plt.imshow(lime_image)
    #     # plt.grid(False)
    #     # plt.show()
    #
    #     # Select the same class
    #     ind = explanation.top_labels[0]
    #     print(ind)
    #
    #     # Map each explanation weight to the corresponding superpixel
    #     dict_heatmap = dict(explanation.local_exp[ind])
    #     heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
    #
    #     # Plot. The visualization makes more sense if a symmetrical colorbar is used.
    #     plt.imshow(heatmap, cmap='RdBu', vmin=-heatmap.max(), vmax=heatmap.max())
    #     plt.colorbar()
    #     plt.grid(False)
    #     # plt.show()
    #     plt.savefig(PATH + EXPORT_FOLDER + modal_name + '/lime_hm/' + str(i) + '_' + str(correct) + str(
    #         np.round(y_pred[i], 2)) + '.png')
    #     plt.clf()
    #
    #     lime_image = cv2.cvtColor(np.array(lime_image * 255, dtype=np.uint8), cv2.COLOR_BGR2RGB)
    #
    #     save_image = np.concatenate((utils.denormalize_x(x_test[i]), lime_image), axis=1)
    #
    #     cv2.imwrite(PATH + EXPORT_FOLDER + modal_name + '/lime/' + str(i) + '_' + str(correct) + str(
    #         np.round(y_pred[i], 2)) + '.png', save_image)

    # # GRAD CAM
    # for i, y in enumerate(y_pred):
    #     img1 = x_test1[i]
    #     img2 = x_test1[i, :, :, 1]
    #     img3 = x_test1[i, :, :, 2]
    #
    #     image_array1 = np.expand_dims(img1, axis=0)
    #     image_array2 = np.expand_dims(img2, axis=0)
    #     image_array3 = np.expand_dims(img3, axis=0)
    #     cam1 = make_gradcam_heatmap([image_array1, image_array2, image_array3], 1, model, 'block5_conv3_m1', 0)
    #     cam2 = make_gradcam_heatmap([image_array1, image_array2, image_array3], 2, model, 'block5_conv3_m2', 0)
    #     cam3 = make_gradcam_heatmap([image_array1, image_array2, image_array3], 3, model, 'block5_conv3_m3', 0)
    #
    #     # f, axarr = plt.subplots(3, 2)
    #     # axarr[0, 0].imshow(utils.denormalize_x(img1), cmap='gray', vmin=0, vmax=255)
    #     # axarr[0, 1].imshow(cam1)
    #     # axarr[1, 0].imshow(utils.denormalize_x(img2), cmap='gray', vmin=0, vmax=255)
    #     # axarr[1, 1].imshow(cam2)
    #     # axarr[2, 0].imshow(utils.denormalize_x(img3), cmap='gray', vmin=0, vmax=255)
    #     # axarr[2, 1].imshow(cam3)
    #
    #     # plt.title(test_names[i])
    #     # plt.show()
    #
    #     correct = 1 if y_pred_binary[i] == y_test[i] else 0
    #
    #     img1 = (img1 + 1) * 127.5
    #     img2 = (img2 + 1) * 127.5
    #     img3 = (img3 + 1) * 127.5
    #     # save_image = np.concatenate((np.repeat(np.expand_dims(img1[:, :, 0], axis=-1), 3, axis=-1),
    #     #                             np.repeat(np.expand_dims(img1[:, :, 1], axis=-1), 3, axis=-1),
    #     #                             np.repeat(np.expand_dims(img1[:, :, 2], axis=-1), 3, axis=-1),
    #     #                             cam1), axis=1)
    #     save_image1 = np.concatenate((img1, cam1), axis=1)
    #     save_image2 = np.concatenate((img2, cam2), axis=1)
    #     save_image3 = np.concatenate((img3, cam3), axis=1)
    #     save_image = np.concatenate((save_image1, save_image2), axis=0)
    #     save_image = np.concatenate((save_image, save_image3), axis=0)
    #
    #     # save_image = np.concatenate((cam1, cam2, cam3), axis=1)
    #
    #     cv2.imwrite(PATH + EXPORT_FOLDER + modal_name + '/cross_val#' + str(CROSS_NUM)
    #                 + '/results/' + str(i) + '_' + str(correct) + str(np.round(y_pred[i], 2)) + '.png', save_image)

    return [roc_auc,
            confusion_matrix[1, 1],
            confusion_matrix[0, 1],
            confusion_matrix[1, 0],
            confusion_matrix[0, 0],
            sensitivity,
            specificity,
            acc]


if __name__ == '__main__':
    print(tf.__version__)
    import matplotlib.pyplot as plt

    # modal_lists = [['tm'], ['am']]

    # modal_lists = [['pc', 'dc'], ['pc', 'tm'], ['pc', 'am'],
    #                ['ec', 'dc'], ['ec', 'tm'], ['ec', 'am'],
    #                ['dc', 'tm'], ['dc', 'am'], ['tm', 'am']]

    modal_lists = [['ec', 'am'],
                   ['dc', 'tm'], ['dc', 'am'], ['tm', 'am']]

    # modal_lists = [['pc', 'ec', 'am'], ['pc', 'ec', 'tm'],
    #                ['pc', 'dc', 'tm'], ['pc', 'dc', 'am'], ['pc', 'tm', 'am'],
    #                ['ec', 'dc', 'tm'], ['ec', 'dc', 'am'], ['ec', 'tm', 'am'],
    #                ['dc', 'tm', 'am']]

    for modal_name in modal_lists:
        print(modal_name)

        # image type = [Img:1, GT:0] , format = {'crop', 'kidney'}, num_classes = [2 / 5]
        x_train_val, y_train_val = load_data_from_xlsx(path=PATH + DATA_FOLDER,
                                                       file_name='data_list2023_train',
                                                       modal_lists=modal_name,
                                                       image_type=1,
                                                       format='crop',
                                                       num_classes=NUM_CLASSES)
        x_test, y_test = load_data_from_xlsx(path=PATH + DATA_FOLDER,
                                             file_name='data_list2023_test',
                                             modal_lists=modal_name,
                                             image_type=1,
                                             format='crop',
                                             num_classes=NUM_CLASSES)

        # y_data, _ = load_data_from_xlsx(PATH + DATA_FOLDER, modal_lists=modal_name, image_type=0)

        # print(x_train_val.shape)
        # print(y_train_val.shape)

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
        # x_train_val, y_train_val, x_test, y_test = split_train_test(x_data, y_data, val_ratio=0.2, random_sample=False)

        if NUM_CLASSES == 2:
            evaluation_results = {'Cross_val': [],
                                  '#Benign': [],
                                  '#Malignant': [],
                                  'AUC': [],
                                  'TP': [],
                                  'FP': [],
                                  'FN': [],
                                  'TN': [],
                                  'Sensitivity': [],
                                  'Specificity': [],
                                  'Accuracy': []}
        else:
            evaluation_results = {'Cross_val': [],
                                  '#AML': [],
                                  '#Oncocytoma': [],
                                  '#CCRCC': [],
                                  '#chromophobeRCC': [],
                                  '#PapillaryRCC': [],
                                  'AUC': [],
                                  'TP': [],
                                  'FP': [],
                                  'FN': [],
                                  'TN': [],
                                  'Sensitivity': [],
                                  'Specificity': [],
                                  'Accuracy': []}

        for CROSS_NUM in range(5):
            print('Cross_val ## ' + str(CROSS_NUM) + '/' + str(5))

            # split train - val
            x_train, y_train, x_val, y_val = split_train_test(x_train_val, y_train_val, val_ratio=0.2, random_sample=False)

            if CROSS_NUM >= 0:
                print('train : #Benign = ' + str(len(y_train[y_train == 0])) + ', #Malignant = ' + str(len(y_train[y_train == 1])))
                print('val : #Benign = ' + str(len(y_val[y_val == 0])) + ', #Malignant = ' + str(len(y_val[y_val == 1])))
                print('test : #Benign = ' + str(len(y_test[y_test == 0])) + ', #Malignant = ' + str(len(y_test[y_test == 1])))

                train(x_train, y_train, x_test, y_test, modal_name)
                results = predict(x_test, y_test, modal_name)

                # collect data
                if NUM_CLASSES == 2:
                    evaluation_results['Cross_val'].append(CROSS_NUM+1)
                    evaluation_results['#Benign'].append(len(y_test[y_test == 0]))
                    evaluation_results['#Malignant'].append(len(y_test[y_test == 1]))
                    evaluation_results['AUC'].append(results[0])
                    evaluation_results['TP'].append(results[1])
                    evaluation_results['FP'].append(results[2])
                    evaluation_results['FN'].append(results[3])
                    evaluation_results['TN'].append(results[4])
                    evaluation_results['Sensitivity'].append(results[5])
                    evaluation_results['Specificity'].append(results[6])
                    evaluation_results['Accuracy'].append(results[7])
                else:
                    evaluation_results['Cross_val'].append(CROSS_NUM+1)
                    evaluation_results['#AML'].append(len(y_test[y_test == 0]))
                    evaluation_results['#Oncocytoma'].append(len(y_test[y_test == 1]))
                    evaluation_results['#CCRCC'].append(len(y_test[y_test == 2]))
                    evaluation_results['#chromophobeRCC'].append(len(y_test[y_test == 3]))
                    evaluation_results['#PapillaryRCC'].append(len(y_test[y_test == 4]))
                    evaluation_results['AUC'].append(results[0])
                    evaluation_results['TP'].append(results[1])
                    evaluation_results['FP'].append(results[2])
                    evaluation_results['FN'].append(results[3])
                    evaluation_results['TN'].append(results[4])
                    evaluation_results['Sensitivity'].append(results[5])
                    evaluation_results['Specificity'].append(results[6])
                    evaluation_results['Accuracy'].append(results[7])

                x_train_val = list_index_move(x_train_val, split_num=0.2)
                y_train_val = list_index_move(y_train_val, split_num=0.2)

        if NUM_CLASSES == 2:
            excel_pixel = pd.DataFrame(evaluation_results, columns=['Cross_val',
                                                                    '#Benign', '#Malignant',
                                                                    'AUC', 'TP', 'FN', 'FP', 'TN',
                                                                    'Sensitivity', 'Specificity', 'Accuracy'])
            excel_pixel.to_excel(PATH + EXPORT_FOLDER + modal_name[0]  + '_' + modal_name[1] + '/evaluation_results.xlsx',
                                 index=None, header=True)
        else:
            excel_pixel = pd.DataFrame(evaluation_results, columns=['Cross_val',
                                                                    '#AML', '#Oncocytoma',
                                                                    '#CCRCC', '#chromophobeRCC', '#PapillaryRCC',
                                                                    'AUC', 'TP', 'FN', 'FP', 'TN',
                                                                    'Sensitivity', 'Specificity', 'Accuracy'])
            excel_pixel.to_excel(PATH + EXPORT_FOLDER + modal_name[0] + '/evaluation_results.xlsx',
                                 index=None, header=True)