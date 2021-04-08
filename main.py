import numpy as np
import glob
import os
import cv2
import tensorflow as tf
import random
import util_function as utils
from myMulmoUnet import MulmoUNet
from loss_function import recall, precision, dice_coef, dice_loss, focal_tv_loss


PATH = '/kw_resource'
DATA_FOLDER = '/datasets/kidney_tumor'
EXPORT_FOLDER = '/exports/mulmo_unet_v1'
IMAGE_SIZE = 256
BATCH_SIZE = 8
EPOCH = 300
CROP_RATIO = 0.15


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


def load_data_from_folder(path, modal_lists, image_type=1):
    # dataset folder structure : Type / patient_id / modal / slice_no

    import os
    # get total files
    total_files = 0
    for root, dirs, files in os.walk(path):
        if root[-2:] + '/' in modal_lists:
            total_files += len(files)

    images = np.zeros((total_files, IMAGE_SIZE, IMAGE_SIZE, 1), np.float32)
    image_names = []
    idx = 0
    for id_name in glob.glob1(path, '*'):
        for modal_name in modal_lists:
            if image_type == 1:   # image
                for slice_name in glob.glob1(path + id_name + '/' + modal_name, '*.png'):
                    image_names.append(id_name + '_' + modal_name[:-1] + '_' + slice_name)
                    image = cv2.imread(path + id_name + '/' + modal_name + slice_name, cv2.IMREAD_GRAYSCALE)
                    image = preprocessing(image, histeq=False)
                    images[idx] = image[:, :, np.newaxis]
                    idx += 1

            elif image_type == 0:   # gt
                for slice_name in glob.glob1(path + id_name + '/' + modal_name, '*.png'):
                    image_names.append(id_name + '_' + modal_name[:-1] + '_' + slice_name)
                    image = cv2.imread(path + id_name + '/' + modal_name[:-1] + 'L/' + slice_name)
                    crop_size = round(image.shape[1]*CROP_RATIO)
                    image = image[:, crop_size:image.shape[1]-crop_size]     # cropping

                    # create mask (red label)
                    mask = np.zeros_like(image[:, :, 0])
                    mask[image[:, :, 2] == 255] = 255
                    mask[image[:, :, 0] > 0] = 0

                    image = np.array(utils.normalize_y(cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE))), np.float32)
                    images[idx] = image[:, :, np.newaxis]
                    idx += 1

    return images, image_names


# data spliting function
def split_train_test(samples, val_ratio=0.2, random_sample=False):
    if random_sample:
        random.shuffle(samples)

    split_idx = int(round(len(samples) * val_ratio))  # split index
    test = samples[:split_idx]
    train = samples[split_idx:]
    return train, test


def train(xy_train, xy_val):

    # unzip input/gt
    x_train, y_train = zip(*xy_train)
    x_val, y_val = zip(*xy_val)
    x_train1 = np.expand_dims(np.asarray(x_train)[:, :, :, 0], axis=3)
    x_train2 = np.expand_dims(np.asarray(x_train)[:, :, :, 1], axis=3)
    x_train3 = np.expand_dims(np.asarray(x_train)[:, :, :, 2], axis=3)
    y_train = np.expand_dims(np.asarray(y_train)[:, :, :, 0], axis=3)
    x_val1 = np.expand_dims(np.asarray(x_val)[:, :, :, 0], axis=3)
    x_val2 = np.expand_dims(np.asarray(x_val)[:, :, :, 1], axis=3)
    x_val3 = np.expand_dims(np.asarray(x_val)[:, :, :, 2], axis=3)
    y_val = np.expand_dims(np.asarray(y_val)[:, :, :, 0], axis=3)

    # load U-Net network
    network = MulmoUNet(1, 1, 32)
    model = network.get_model()
    model.summary()

    model.compile(loss=focal_tv_loss(0.7, 2.0),
                  optimizer=tf.optimizers.Adam(learning_rate=0.001),
                  metrics=[dice_coef, recall, precision])

    # data augmentation
    aug = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=5,
                                                          width_shift_range=0.1,
                                                          height_shift_range=0.1,
                                                          zoom_range=False,
                                                          fill_mode='nearest',
                                                          vertical_flip=False,
                                                          horizontal_flip=True)

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

    # learning rate decay callback
    lr_reduce_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                              mode='min',
                                                              factor=0.5,
                                                              patience=10,
                                                              min_lr=5e-5)

    # model checkpoint
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=PATH + '/weight_checkpoint.hdf5',
                                                                   save_weights_only=True,
                                                                   monitor='val_loss',
                                                                   mode='min',
                                                                   save_best_only=True)

    # fits the model on batches with real-time data augmentation:
    history = model.fit(generator(x_train1, x_train2, x_train3, y_train, batch_size=BATCH_SIZE),
                        steps_per_epoch=len(x_train) / BATCH_SIZE,
                        epochs=EPOCH,
                        callbacks=[lr_reduce_callback, model_checkpoint_callback],
                        validation_data=([x_val1, x_val2, x_val3], y_val))

    model.save_weights(PATH + EXPORT_FOLDER + '/mulmounet_weights.hdf5')
    utils.plot_summary_graph(history, PATH + EXPORT_FOLDER)


def predict(xy_test):

    def pixelbased_metric(y_true, y_pred, binary_th=0.5):
        _, y_true = cv2.threshold(y_true, binary_th * 255, 255, cv2.THRESH_BINARY)
        _, y_pred = cv2.threshold(y_pred, binary_th * 255, 255, cv2.THRESH_BINARY)

        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        tp = np.sum(np.logical_and(y_true == 255, y_pred == 255))
        fp = np.sum(np.logical_and(y_true == 0, y_pred == 255))
        fn = np.sum(np.logical_and(y_true == 255, y_pred == 0))

        return tp, fn, fp

    def metrics_compute(tp, fn, fp):
        def recall_compute(tp, fn):
            return round((tp * 100)/(tp + fn), 2) if tp + fn > 0 else 0

        def precision_compute(tp, fp):
            return round((tp * 100)/(tp + fp), 2) if tp + fp > 0 else 0

        def f_score_compute(tp, fn, fp):
            a = recall_compute(tp, fn) * precision_compute(tp, fp)
            b = recall_compute(tp, fn) + precision_compute(tp, fp)
            return round((a/b), 2) if b > 0 else 0

        return recall_compute(tp, fn), precision_compute(tp, fp), f_score_compute(tp, fn, fp)

    # unzip input/gt
    x_test, y_test = zip(*xy_test)
    x_test1 = np.expand_dims(np.asarray(x_test)[:, :, :, 0], axis=3)
    x_test2 = np.expand_dims(np.asarray(x_test)[:, :, :, 1], axis=3)
    x_test3 = np.expand_dims(np.asarray(x_test)[:, :, :, 2], axis=3)
    y_test = np.expand_dims(np.asarray(y_test)[:, :, :, 0], axis=3)

    # load U-Net network with trained weights
    network = MulmoUNet(1, 1, 32)
    model = network.get_model()
    model.load_weights(PATH + EXPORT_FOLDER + '/mulmounet_weights.hdf5')

    # predict
    y_preds = model.predict([x_test1, x_test2, x_test3], batch_size=BATCH_SIZE)

    # evaluation (find pixel-wise recall, precision, f1 score)
    recall = []
    precision = []
    f1_score = []
    try:
        os.makedirs(PATH + EXPORT_FOLDER + '/test_images')
    except(FileExistsError):
        print('folders exist')

    for i, y_pred in enumerate(y_preds):
        # denormalize (convert to 8-bit grayscale)
        y_true = np.array(utils.denormalize_y(y_test[i, :, :, 0]), dtype=np.uint8)
        y_pred = np.array(utils.denormalize_y(y_pred), dtype=np.uint8)

        # save heatmap comparison
        org_image = utils.denormalize_x(x_test1[i, :, :, 0])
        y_true_heatmap = utils.convert_to_heatmap(org_image, y_true)
        y_pred_heatmap = utils.convert_to_heatmap(org_image, y_pred)
        save_image = np.concatenate((y_true_heatmap, y_pred_heatmap), axis=1)
        cv2.imwrite(PATH + EXPORT_FOLDER + '/test_images/' + str(i) + '.png', save_image)

        # pixel-based
        tp, fn, fp = pixelbased_metric(y_true, y_pred, binary_th=0.5)

        recall.append(metrics_compute(tp, fn, fp)[0])
        precision.append(metrics_compute(tp, fn, fp)[1])
        f1_score.append(metrics_compute(tp, fn, fp)[2])

    def average(lst):
        return sum(lst) / len(lst)

    print('Recall: ' + str(average(recall)))
    print('Precision: ' + str(average(precision)))
    print('F1 score: ' + str(average(f1_score)))


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # modal = ['pc/', 'ec/', 'dc/']  # 'pc/', 'ec/', 'dc/', 'tm/', 'am/'

    x0_data1, _ = load_data_from_folder(DATA_FOLDER + '/AML/', modal_lists=['pc/'], image_type=1)
    y0_data1, _ = load_data_from_folder(DATA_FOLDER + '/AML/', modal_lists=['pc/'], image_type=0)
    x1_data1, _ = load_data_from_folder(DATA_FOLDER + '/CCRCC/', modal_lists=['pc/'], image_type=1)
    y1_data1, _ = load_data_from_folder(DATA_FOLDER + '/CCRCC/', modal_lists=['pc/'], image_type=0)

    x0_data2, _ = load_data_from_folder(DATA_FOLDER + '/AML/', modal_lists=['ec/'], image_type=1)
    y0_data2, _ = load_data_from_folder(DATA_FOLDER + '/AML/', modal_lists=['ec/'], image_type=0)
    x1_data2, _ = load_data_from_folder(DATA_FOLDER + '/CCRCC/', modal_lists=['ec/'], image_type=1)
    y1_data2, _ = load_data_from_folder(DATA_FOLDER + '/CCRCC/', modal_lists=['ec/'], image_type=0)

    x0_data3, _ = load_data_from_folder(DATA_FOLDER + '/AML/', modal_lists=['dc/'], image_type=1)
    y0_data3, _ = load_data_from_folder(DATA_FOLDER + '/AML/', modal_lists=['dc/'], image_type=0)
    x1_data3, _ = load_data_from_folder(DATA_FOLDER + '/CCRCC/', modal_lists=['dc/'], image_type=1)
    y1_data3, _ = load_data_from_folder(DATA_FOLDER + '/CCRCC/', modal_lists=['dc/'], image_type=0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # display sample images
    f, axarr = plt.subplots(3, 3)
    idx = random.randint(0, len(x0_data1))
    axarr[0, 0].imshow(utils.denormalize_x(x0_data1[idx]), cmap='gray', vmin=0, vmax=255)
    axarr[0, 1].imshow(utils.denormalize_x(x0_data2[idx]), cmap='gray', vmin=0, vmax=255)
    axarr[0, 2].imshow(utils.denormalize_x(x0_data3[idx]), cmap='gray', vmin=0, vmax=255)
    axarr[1, 0].imshow(clahe.apply(utils.denormalize_x(x0_data1[idx])), cmap='gray', vmin=0, vmax=255)
    axarr[1, 1].imshow(clahe.apply(utils.denormalize_x(x0_data2[idx])), cmap='gray', vmin=0, vmax=255)
    axarr[1, 2].imshow(clahe.apply(utils.denormalize_x(x0_data3[idx])), cmap='gray', vmin=0, vmax=255)
    axarr[2, 0].imshow(utils.denormalize_y(y0_data1[idx]), cmap='gray', vmin=0, vmax=255)
    axarr[2, 1].imshow(utils.denormalize_y(y0_data2[idx]), cmap='gray', vmin=0, vmax=255)
    axarr[2, 2].imshow(utils.denormalize_y(y0_data3[idx]), cmap='gray', vmin=0, vmax=255)
    # plt.show()

    x_data = np.concatenate(([np.concatenate([x0_data1, x1_data1]),
                              np.concatenate([x0_data2, x1_data2]),
                              np.concatenate([x0_data3, x1_data3])]), axis=3)
    y_data = np.concatenate(([np.concatenate([y0_data1, y1_data1]),
                              np.concatenate([y0_data2, y1_data2]),
                              np.concatenate([y0_data3, y1_data3])]), axis=3)

    # split train - test
    xy_train, xy_test = split_train_test(list(zip(x_data, y_data)), val_ratio=0.2)

    print(tf.__version__)

    print('# train: ' + str(len(xy_train)))
    print('# test: ' + str(len(xy_test)))
    train(xy_train, xy_test)
    predict(xy_test)

