import numpy as np
import cv2 as cv
import os

###############################################################
##### This code has been tested in Python 3.6 environment #####
###############################################################

def cal_result(datapath, resultpath):

    ##### groundtruth image
    groundtruth = [img for img in sorted(os.listdir(datapath)) if img.endswith(".png")]

    ##### your result image
    result_image = [img for img in sorted(os.listdir(resultpath)) if img.endswith(".png")]

    assert len(groundtruth) == len(result_image), "Result should have the same number of samples as groundtruth"

    ##### variables for confusion matrix
    TP = 0   # True Positive
    TN = 0   # True Negative
    FP = 0   # False Positive
    FN = 0   # False Negative
    GT = 0   # Number of GroundTruth Pixels
    EST = 0  # Number of Estimated Pixels

    ##### make confusion matrix for images
    for index in range(len(groundtruth)):

        ##### groundtruth pre-processing
        gt = cv.imread(os.path.join(datapath, groundtruth[index]))
        gt_gray = cv.cvtColor(gt, cv.COLOR_BGR2GRAY)
        gt_cvt = np.where(gt_gray == 255, 255, 0)
        gt_cvt_rev = np.where(gt_gray == 0, 255, 0)

        ##### result pre-processing
        result = cv.imread(os.path.join(resultpath, result_image[index]))
        result_gray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)

        ##### variable summation
        TP += np.sum(np.logical_and(gt_cvt, result_gray).astype(np.uint8))
        FP += np.sum(np.logical_and(gt_cvt_rev, result_gray).astype(np.uint8))
        GT += np.sum(gt_cvt) // 255

    print('TP : %d  ||  GT : %d ||  EST : %d' % (TP, GT, TP + FP))
    print('Recall : %.3f %%  ||  Precision : %.3f %%' %
          (((100.0 * TP) / GT), ((100.0 * TP) / (TP + FP))))


if __name__ == '__main__':
    cal_result('./groundtruth', './result')