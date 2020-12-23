import numpy as np
import cv2 as cv
import os
import evaluation as eval

###############################################################
##### This code has been tested in Python 3.6 environment #####
###############################################################

def main():

    ##### Set threshold
    threshold = 20
    
    ##### Set path
    input_path = './input_image'    # input path
    gt_path = './groundtruth'       # groundtruth path
    result_path = './result'        # result path

    ##### load input
    input = [img for img in sorted(os.listdir(input_path)) if img.endswith(".jpg")]

    ##### first frame and first background
    frame_current = cv.imread(os.path.join(input_path, input[0]))
    frame_current_gray = cv.cvtColor(frame_current, cv.COLOR_BGR2GRAY).astype(np.float64)
    frame_prev_gray = frame_current_gray


    ##### background substraction
    for image_idx in range(len(input)):

        ##### calculate foreground region
        diff = frame_current_gray - frame_prev_gray
        diff_abs = np.abs(diff).astype(np.float64)

        ##### make mask by applying threshold
        frame_diff = np.where(diff_abs > threshold, 1.0, 0.0)

        ##### apply mask to current frame
        current_gray_masked = np.multiply(frame_current_gray, frame_diff)
        current_gray_masked_mk2 = np.where(current_gray_masked > 0, 255.0, 0.0)

        ##### final result
        result = current_gray_masked_mk2.astype(np.uint8)
        cv.imshow('result', result)

        ##### renew background
        frame_prev_gray = frame_current_gray

        ##### make result file
        ##### Please don't modify path
        cv.imwrite(os.path.join(result_path, 'result%06d.png' % (image_idx + 1)), result)

        ##### end of input
        if image_idx == len(input) - 1:
            break

        ##### read next frame
        frame_current = cv.imread(os.path.join(input_path, input[image_idx + 1]))
        frame_current_gray = cv.cvtColor(frame_current, cv.COLOR_BGR2GRAY).astype(np.float64)

        ##### If you want to stop, press ESC key
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break

    ##### evaluation result
    eval.cal_result(gt_path, result_path)

if __name__ == '__main__':
    main()

