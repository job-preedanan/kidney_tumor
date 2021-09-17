import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
import os


def cropping(image, gt):
    bw_gt = np.zeros_like(gt[:, :, 0])
    bw_gt[gt[:, :, 2] == 255] = 255
    bw_gt[gt[:, :, 0] > 0] = 0
    tumor_cnt = cv2.findContours(bw_gt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(tumor_cnt[0][0])
    b = round(0.2*h) if h > w else round(0.2*w)

    # cv2.rectangle(gt, (x - b, y - b), (x + w + b, y + h + b), (0, 0, 255), 3)
    crop = image[y-b:y+h+b, x-b:x+w+b]

    # plt.imshow(crop, cmap='gray')
    # plt.show()

    return crop


def create_cropped_tumor_dataset(path, path_save):
    for id_name in glob.glob1(path, '*'):
        for modal_name in glob.glob1(path + id_name, '*'):
            os.makedirs(path_save + id_name + '/' + modal_name)
            for slice_name in glob.glob1(path + id_name + '/' + modal_name, '*.png'):
                try:
                    image = cv2.imread(path + id_name + '/' + modal_name + '/' + slice_name, cv2.IMREAD_GRAYSCALE)
                    gt = cv2.imread(path + id_name + '/' + modal_name + 'L/' + slice_name)
                    gt = cv2.resize(gt, (image.shape[1], image.shape[0]))
                    crop = cropping(image, gt)
                    cv2.imwrite(path_save + id_name + '/' + modal_name + '/' + slice_name, crop)
                    print(id_name + '/' + modal_name + '/' + slice_name)
                except:
                    print(id_name, modal_name)


if __name__ == '__main__':
    path = 'C:/Users/Job/Documents/DoctorProject/kidney_tumor/dataset/AML/'
    path_save = 'C:/Users/Job/Documents/DoctorProject/kidney_tumor/dataset/AML_crop/'
    create_cropped_tumor_dataset(path, path_save)

