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
    avg_path = './average'          #average_path
    ncp_path = './NCP'
    basic_path = './'

    ##### load input
    input = [img for img in sorted(os.listdir(input_path)) if img.endswith(".jpg")]

    ##### first frame and first background
    frame_current = cv.imread(os.path.join(input_path, input[0]))
    frame_current_gray = cv.cvtColor(frame_current, cv.COLOR_BGR2GRAY).astype(np.float64)
    frame_prev_gray = frame_current_gray

    frame_gray_sum = frame_prev_gray
    for image_idx in range(1,len(input)):
        frame_current = cv.imread(os.path.join(input_path, input[image_idx]))
        frame_current_gray = cv.cvtColor(frame_current, cv.COLOR_BGR2GRAY).astype(np.float64)
        
        frame_gray_sum += frame_current_gray

    frame_gray_avg = frame_gray_sum / (len(input))
    frame_gray_avg.astype(np.uint8)
    # cv.imshow('Average', frame_gray_avg)
    cv.imwrite(os.path.join(basic_path, 'average.png'), frame_gray_avg)
    ###Will STD(standard deviation) help it???

    
    #************minus average**************#

    for image_idx in range(len(input)):

        frame_current = cv.imread(os.path.join(input_path, input[image_idx]))
        frame_current_gray = cv.cvtColor(frame_current, cv.COLOR_BGR2GRAY).astype(np.float64)

        avg_diff = frame_current_gray - frame_gray_avg
        
        avg_diff_abs = np.abs(avg_diff).astype(np.float64)

        ##### make mask by applying threshold
        avg_frame_diff = np.where(avg_diff_abs > 40, 0.5, -0.5)

        ##### apply mask to current frame
        avg_current_gray_masked = np.multiply(frame_current_gray, avg_frame_diff)
        avg_current_gray_masked_mk2 = np.where(avg_current_gray_masked > 0, 255.0, 0.0)

        ##### final result
        avg_result = avg_current_gray_masked_mk2.astype(np.uint8)
        # cv.imshow('avg_result', avg_result)

        ##### make result file
        ##### Please don't modify path
        cv.imwrite(os.path.join(avg_path, 'avg_result%06d.png' % (image_idx + 1)), avg_result)

        ##### end of input
        if image_idx == len(input) - 1:
            break
    print("Minus Average")
    eval.cal_result(gt_path, avg_path)
    print("")

    #*********************next-current-previous************************************************#
    
    ncp_frame_current = cv.imread(os.path.join(input_path, input[0]))
    ncp_frame_current_gray = cv.cvtColor(ncp_frame_current, cv.COLOR_BGR2GRAY).astype(np.float64)
    ncp_frame_next = cv.imread(os.path.join(input_path, input[2]))
    ncp_frame_next_gray = cv.cvtColor(ncp_frame_next, cv.COLOR_BGR2GRAY).astype(np.float64)
    ncp_frame_prev_gray = ncp_frame_current_gray
    ncp_frame_current = cv.imread(os.path.join(input_path, input[1]))
    ncp_frame_current_gray = cv.cvtColor(ncp_frame_current, cv.COLOR_BGR2GRAY).astype(np.float64)


    diff_CP = ncp_frame_current_gray - ncp_frame_prev_gray
    diff_CP_abs = np.abs(diff_CP).astype(np.float64)
    ncp_frame_diff = np.where(diff_CP_abs > 60, 0.1, -0.2)
    ncp_current_gray_masked = np.multiply(ncp_frame_current_gray, ncp_frame_diff)
    ncp_current_gray_masked_mk2 = np.where(ncp_current_gray_masked > 0, 255.0, 0.0)
    ncp_result = ncp_current_gray_masked_mk2.astype(np.uint8)
    cv.imwrite(os.path.join(ncp_path, 'ncp_result%06d.png' % (1)), ncp_result)


    ##### background substraction
    for image_idx in range(1,len(input)):

        ##### calculate foreground region
        diff_CP = ncp_frame_current_gray - ncp_frame_prev_gray
        diff_NC = ncp_frame_next_gray - ncp_frame_current_gray
        diff_CP_abs = np.abs(diff_CP).astype(np.float64)
        diff_NC_abs = np.abs(diff_NC).astype(np.float64)

        diff_abs = (diff_CP_abs + diff_NC_abs) / 2
        if image_idx == len(input) - 1:
            diff_abs *= 2

        ##### make mask by applying threshold
        ncp_frame_diff = np.where(diff_abs > 35, 0.1, -0.2)

        ##### apply mask to current frame
        ncp_current_gray_masked = np.multiply(ncp_frame_current_gray, ncp_frame_diff)
        ncp_current_gray_masked_mk2 = np.where(ncp_current_gray_masked > 0, 255.0, 0.0)

        ##### final result
        ncp_result = ncp_current_gray_masked_mk2.astype(np.uint8)
        # cv.imshow('ncp_result', ncp_result)

        ##### renew background
        ncp_frame_prev_gray = ncp_frame_current_gray

        ##### make result file
        ##### Please don't modify path
        cv.imwrite(os.path.join(ncp_path, 'ncp_result%06d.png' % (image_idx + 1)), ncp_result)

        ##### end of input
        if image_idx == len(input) - 1:
            break
        if image_idx == len(input) - 2:
            ncp_frame_current = cv.imread(os.path.join(input_path, input[image_idx + 1]))
            ncp_frame_current_gray = cv.cvtColor(ncp_frame_current, cv.COLOR_BGR2GRAY).astype(np.float64)
            ncp_frame_next_gray = ncp_frame_current_gray
            continue

            
        ##### read next frame
        ncp_frame_current = cv.imread(os.path.join(input_path, input[image_idx + 1]))
        ncp_frame_current_gray = cv.cvtColor(ncp_frame_current, cv.COLOR_BGR2GRAY).astype(np.float64)

        ncp_frame_next = cv.imread(os.path.join(input_path, input[image_idx + 2]))
        ncp_frame_next_gray = cv.cvtColor(ncp_frame_next, cv.COLOR_BGR2GRAY).astype(np.float64)

        

        ##### If you want to stop, press ESC key
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break

    ##### evaluation result
    print("Next - Current - Previous")
    eval.cal_result(gt_path, ncp_path)
    print("")




    
    
    #**********************Basic One***************************#
    frame_current = cv.imread(os.path.join(input_path, input[0]))
    frame_current_gray = cv.cvtColor(frame_current, cv.COLOR_BGR2GRAY).astype(np.float64)
    frame_prev_gray = frame_current_gray

    ##### background substraction
    for image_idx in range(len(input)):

        ##### calculate foreground region
        diff = frame_current_gray - frame_prev_gray
        diff_abs = np.abs(diff).astype(np.float64)

        ##### make mask by applying threshold
        # frame_diff = np.where(diff_abs > threshold, 1.0, 0.0)

        frame_diff = np.where(diff_abs > 10, 0.1, -0.2)
        

        ### Yeah I'm an idiot, 

        ##### apply mask to current frame
        current_gray_masked = np.multiply(frame_current_gray, frame_diff)
        current_gray_masked_mk2 = np.where(current_gray_masked > 0, 255.0, 0.0)

        ##### final result
        result = current_gray_masked_mk2.astype(np.uint8)
        # cv.imshow('result', result)

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
    print("Basic Baseline")
    eval.cal_result(gt_path, result_path)

if __name__ == '__main__':
    main()

