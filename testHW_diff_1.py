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
    avg_path = './average'          # average_path
    ncp_path = './NCP'              # ncp path
    mtg_path = './MTG'              # mtg path
    std_path = './STD'              # std path
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
    

    #************minus average**************#

    for image_idx in range(len(input)):

        frame_current = cv.imread(os.path.join(input_path, input[image_idx]))
        frame_current_gray = cv.cvtColor(frame_current, cv.COLOR_BGR2GRAY).astype(np.float64)

        avg_diff = frame_current_gray - frame_gray_avg
        
        avg_diff_abs = np.abs(avg_diff).astype(np.float64)

        ##### make mask by applying threshold
        avg_frame_diff = np.where(avg_diff_abs > 40, 1.0, 0.0)

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

    #************minus average and std**************#
    ###Will STD(standard deviation) help it???
    white_template = cv.imread(os.path.join(basic_path, 'white_template.png'))
    white_template_gray = cv.cvtColor(white_template, cv.COLOR_BGR2GRAY).astype(np.float64)
    std_frame_current = cv.imread(os.path.join(input_path, input[0]))
    std_frame_current_gray = cv.cvtColor(std_frame_current, cv.COLOR_BGR2GRAY).astype(np.float64)
    std_diff = std_frame_current_gray - frame_gray_avg
    std_diff_square = std_diff * std_diff
    std_tmps = std_diff_square
    for image_idx in range(1,len(input)):
        std_frame_current = cv.imread(os.path.join(input_path, input[image_idx]))
        std_frame_current_gray = cv.cvtColor(std_frame_current, cv.COLOR_BGR2GRAY).astype(np.float64)
        std_diff = std_frame_current_gray - frame_gray_avg
        std_diff_square = np.square(std_diff)
        std_tmps += std_diff_square
    std_std = np.sqrt(std_tmps)
    rev_std_std = np.abs(white_template_gray-std_std).astype(np.float64)
    cv.imwrite(os.path.join(basic_path, 'Standard Deviation.png'), std_std)
    cv.imwrite(os.path.join(basic_path, 'Reversed Standard Deviation.png'), rev_std_std)
    

    for image_idx in range(len(input)):

        std_frame_current = cv.imread(os.path.join(input_path, input[image_idx]))
        std_frame_current_gray = cv.cvtColor(std_frame_current, cv.COLOR_BGR2GRAY).astype(np.float64)

        std_diff = std_frame_current_gray - frame_gray_avg
        
        std_diff_plus = std_diff + std_std
        std_diff_minus = std_diff - std_std

        std_diff_abs_pl = np.abs(std_diff_plus).astype(np.float64)
        std_diff_abs_mi = np.abs(std_diff_minus).astype(np.float64)
        


        ##### make mask by applying threshold
        std_frame_diff_pl = np.where(std_diff_abs_pl > 40, 1.0, 0.0)
        std_frame_diff_mi = np.where(std_diff_abs_mi > 40, 1.0, 0.0)
        

        ##### apply mask to current frame
        std_current_gray_masked_pl = np.multiply(std_frame_current_gray, std_frame_diff_pl)
        std_current_gray_masked_mi = np.multiply(std_frame_current_gray, std_frame_diff_mi)
        
        # std_current_gray_masked_mk2_1 = np.where(std_current_gray_masked_pl > 0, 127.5, -127.5)
        # std_current_gray_masked_mk2_2 = np.where(std_current_gray_masked_mi > 0, 127.5, -127.5)

        # std_current_gray_masked_mk2 = np.where(std_current_gray_masked_pl > 0, 255.0, 0.0)
        std_current_gray_masked_mk2 = np.where(std_current_gray_masked_mi > 0, 255.0, 0.0)

        

        # std_current_gray_masked_mk2 = std_current_gray_masked_mk2_1 + std_current_gray_masked_mk2_2
        


        ##### final result
        std_result = std_current_gray_masked_mk2.astype(np.uint8)
        # cv.imshow('avg_result', avg_result)

        ##### make result file
        ##### Please don't modify path
        cv.imwrite(os.path.join(std_path, 'std_result%06d.png' % (image_idx + 1)), std_result)

        ##### end of input
        if image_idx == len(input) - 1:
            break
    print("Minus Average And STD")
    eval.cal_result(gt_path, std_path)
    print("")
    ### Nah It Didn't help it


    ### Car Trail ###
    CT_frame_current = cv.imread(os.path.join(input_path, input[0]))
    CT_frame_current_gray = cv.cvtColor(CT_frame_current, cv.COLOR_BGR2GRAY).astype(np.float64)
    CT_frame_gray_accm = CT_frame_current_gray - frame_gray_avg

    for image_idx in range(1,len(input)):
        CT_frame_current = cv.imread(os.path.join(input_path, input[image_idx]))
        CT_frame_current_gray = cv.cvtColor(CT_frame_current, cv.COLOR_BGR2GRAY).astype(np.float64)
        CT_frame_gray_diff = CT_frame_current_gray - frame_gray_avg
        CT_frame_gray_accm += np.abs(CT_frame_gray_diff).astype(np.float64)

    CT_result = (CT_frame_gray_accm/len(input)).astype(np.uint8)
    cv.imwrite(os.path.join(basic_path, 'CT_float64.png'), CT_frame_gray_accm/len(input))
    cv.imwrite(os.path.join(basic_path, 'CT_int8.png'), CT_result)

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
    ncp_frame_diff = np.where(diff_CP_abs > 50, 0.1, -0.2)
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
        # ncp_frame_diff_CP = np.where(diff_CP_abs > 5, 0.1, -0.2)
        # ncp_frame_diff_NC = np.where(diff_NC_abs > 50, 0.1, -0.2)

        
        # ncp_frame_diff = ncp_frame_diff_CP + ncp_frame_diff_NC
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

        frame_diff = np.where(diff_abs > 65, 1.0, 0.0)
        

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
    print("")

    #**********************MinTheGap***************************#
    MTG_frame_current = cv.imread(os.path.join(input_path, input[0]))
    MTG_frame_current_gray = cv.cvtColor(MTG_frame_current, cv.COLOR_BGR2GRAY).astype(np.float64)
    # MTG_frame_prev_gray = MTG_frame_current_gray

    ##### background substraction
    for image_idx in range(len(input)):

        ##### calculate foreground region
        MTG_diff = MTG_frame_current_gray - frame_gray_avg
        MTG_diff_abs = np.abs(MTG_diff).astype(np.float64)

        ##### make mask by applying threshold
        # frame_diff = np.where(diff_abs > threshold, 1.0, 0.0)

        MTG_frame_diff = np.where(MTG_diff_abs > 80, 1.0, 0.0)

        ##### apply mask to current frame
        MTG_current_gray_masked = np.multiply(MTG_frame_current_gray, MTG_frame_diff)
        MTG_current_gray_masked_mk2 = np.where(MTG_current_gray_masked > 0, 255.0, 0.0)
        # print("len(MTG_current_gray_masked_mk2)")
        # print(len(MTG_current_gray_masked_mk2))
        # print("")
        for i in range(len(MTG_current_gray_masked_mk2)):
            tmp_indexes=[]
            if list(MTG_current_gray_masked_mk2[i]).count(255.0) > 1:
                # print("index")
                # print(i)
                # print("")
                # print("list(MTG_current_gray_masked_mk2[i]).count(255.0)")
                # print(list(MTG_current_gray_masked_mk2[i]).count(255.0))
                # print("")

                tmp_mk2_list = MTG_current_gray_masked_mk2[i]
                for j in range(list(MTG_current_gray_masked_mk2[i]).count(255.0)):
                    tmpInd = list(tmp_mk2_list).index(255.0)
                    tmp_indexes.append(tmpInd)
                    tmp_mk2_list = MTG_current_gray_masked_mk2[i][tmpInd+1:]
                for k in range(1,len(tmp_indexes)):
                    if tmp_indexes[k] - tmp_indexes[k-1] < 20: # (len(MTG_current_gray_masked_mk2[i])/10):
                        if tmp_indexes[k] - tmp_indexes[k-1] != 1:
                            for l in range(tmp_indexes[k-1],tmp_indexes[k]):
                                MTG_current_gray_masked_mk2[i][l] = 255.0
        

        # print("MTG_current_gray_masked_mk2")
        # print(MTG_current_gray_masked_mk2)
        # print("")

        

        ##### final result
        MTG_result = MTG_current_gray_masked_mk2.astype(np.uint8)
        # cv.imshow('result', result)

        ##### renew background
        # MTG_frame_prev_gray = MTG_frame_current_gray

        ##### make result file
        ##### Please don't modify path
        cv.imwrite(os.path.join(mtg_path, 'mtg_result%06d.png' % (image_idx + 1)), MTG_result)

        ##### end of input
        if image_idx == len(input) - 1:
            break

        ##### read next frame
        MTG_frame_current = cv.imread(os.path.join(input_path, input[image_idx + 1]))
        MTG_frame_current_gray = cv.cvtColor(MTG_frame_current, cv.COLOR_BGR2GRAY).astype(np.float64)

        ##### If you want to stop, press ESC key
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break

    ##### evaluation result
    print("Mind The Gap")
    eval.cal_result(gt_path, mtg_path)
    print("")

if __name__ == '__main__':
    main()

