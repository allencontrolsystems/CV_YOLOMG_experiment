import cv2
import os
import numpy as np
from MOD_Functions import motion_compensate
from MOD_Functions import enlargebox
import imgviz

kernel_size = 3


def FD3_mask(lastFrame1, lastFrame2, currentFrame, video_name, frame_count, save_path=None):
    lastFrame1 = cv2.GaussianBlur(lastFrame1, (11, 11), 0)
    lastFrame1 = cv2.cvtColor(lastFrame1, cv2.COLOR_BGR2GRAY)

    lastFrame2 = cv2.GaussianBlur(lastFrame2, (11, 11), 0)
    lastFrame2 = cv2.cvtColor(lastFrame2, cv2.COLOR_BGR2GRAY)

    currentFrame = cv2.GaussianBlur(currentFrame, (11, 11), 0)
    currentFrame = cv2.cvtColor(currentFrame, cv2.COLOR_BGR2GRAY)

    img_compensate1, mask1, avg_dist1, motion_x1, motion_y1, homo_matrix = motion_compensate(lastFrame1, lastFrame2)
    frameDiff1 = cv2.absdiff(lastFrame2, img_compensate1)
    # fix_coef1 = np.mean(frameDiff1)
    # fix_coef1 = int(fix_coef1)
    # T_1 = 4 + fix_coef1
    # _, thresh1 = cv2.threshold(frameDiff1, T_1, 255, cv2.THRESH_BINARY)
    # thresh1 = thresh1 - mask1
    # thresh1 = cv2.medianBlur(thresh1, 3)

    img_compensate2, mask2, avg_dist2, motion_x2, motion_y2, homo_matrix2 = motion_compensate(currentFrame, lastFrame2)
    frameDiff2 = cv2.absdiff(lastFrame2, img_compensate2)
    # fix_coef2 = np.mean(frameDiff2)
    # fix_coef2 = int(fix_coef2)
    # T_2 = 4 + fix_coef2
    # _, thresh2 = cv2.threshold(frameDiff2, T_2, 255, cv2.THRESH_BINARY)
    # thresh2 = thresh2 - mask2
    # thresh2 = cv2.medianBlur(thresh2, 3)
    #
    # thresh = cv2.bitwise_or(thresh1, thresh2)

    frameDiff = (frameDiff1 + frameDiff2) / 2

    # _, thresh3 = cv2.threshold(np.uint8(frameDiff), 5, 255, cv2.THRESH_BINARY)
    # thresh3 = thresh3 - mask1
    # thresh = thresh3 - mask2
    #
    # width = frameDiff.shape[1]
    # height = frameDiff.shape[0]
    #
    # if width == 1920:
    #     frameDiff[600:750, 0:320] = 0
    # else:
    #     frameDiff[500:650, 0:240] = 0

    # if width == 1920:
    #     thresh[620:720, 0:240] = 0
    # else:
    #     thresh[540:640, 0:140] = 0
    # frame_depth = frameDiff.astype(np.float)
    # frame_viz = imgviz.depth2rgb(frame_depth, min_value=5, max_value=30)

    # 对阈值图像进行开操作，减少噪声
    # kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    # open_demo = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel1)
    #
    # # 对开操作之后的图像做闭操作，减少孔洞，填充空隙
    # kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    # close_demo = cv2.morphologyEx(open_demo, cv2.MORPH_CLOSE, kernel2, iterations=3)
    # cv2.imshow('Morphological Operation', close_demo)

    # save_path = '/home/user-guo/data/drone-dataset/phantom-dataset/mask31/' + video_name

    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # cv2.imwrite(save_path + '/' + video_name + '_' + str(frame_count).zfill(4) + '.jpg', frameDiff)
    
    # if save_path is None:
    #     save_path = '/home/acs/YOLOMG/full_data/phantom-dataset/mask31/' + video_name
    #
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # cv2.imwrite(save_path + '/' + video_name + '_' + str(frame_count).zfill(4) + '.jpg', frameDiff)

    # contours, hierarchy = cv2.findContours(close_demo.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #
    # # traverse contours
    # rect_list = []
    # for contour in contours:
    #     # if contour is too small or too big, ignore it
    #     (x, y, w, h) = cv2.boundingRect(contour)
    #     # area = cv2.contourArea(contour)
    #     area = w * h
    #     ratio = w / h
    #     if 25 < area < 3000 and 0.3 < ratio < 3.0:
    #         rect = (x, y, w, h)
    #         rect_list.append(rect)
    #
    # # rect_merge = box_select(np.array(rect_list))
    # rect_merge = rect_list
    # obj_num = len(rect_merge)

    return frameDiff
