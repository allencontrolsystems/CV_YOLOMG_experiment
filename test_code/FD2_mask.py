import cv2
import os
import numpy as np
from MOD_Functions import motion_compensate
from MOD_Functions import enlargebox

kernel_size = 3


def FD2_mask(lastFrame1, lastFrame2, frame_count, video_name, save_location=None):
    lastFrame1 = cv2.GaussianBlur(lastFrame1, (11, 11), 0)
    lastFrame1 = cv2.cvtColor(lastFrame1, cv2.COLOR_BGR2GRAY)

    lastFrame2 = cv2.GaussianBlur(lastFrame2, (11, 11), 0)
    lastFrame2 = cv2.cvtColor(lastFrame2, cv2.COLOR_BGR2GRAY)

    img_compensate1, mask1, avg_dist1, motion_x1, motion_y1, homo_matrix = motion_compensate(lastFrame1, lastFrame2)  # version2
    frameDiff1 = cv2.absdiff(lastFrame2, img_compensate1)
    # fix_coef1 = np.mean(frameDiff1)
    # fix_coef1 = int(fix_coef1)
    # T_1 = 5 + fix_coef1
    # _, thresh1 = cv2.threshold(frameDiff1, T_1, 255, cv2.THRESH_BINARY)
    # thresh = thresh1 - mask1
    # thresh = cv2.medianBlur(thresh, 5)

    # 对阈值图像进行开操作，减少噪声
    # kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    # open_demo = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel1)
    #
    # # 对开操作之后的图像做闭操作，减少孔洞，填充空隙
    # kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    # close_demo = cv2.morphologyEx(open_demo, cv2.MORPH_CLOSE, kernel2, iterations=3)
    # cv2.imshow('Morphological Operation', close_demo)

    if save_location is None:
        save_path = '/home/user-guo/data/drone-videos/Drone-vs-Bird/mask22/' + video_name
    else:
        save_path = save_location

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cv2.imwrite(save_path + '/' + video_name + '_' + str(frame_count).zfill(4) + '.jpg', frameDiff1)

    # contours, hierarchy = cv2.findContours(close_demo.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # traverse contours
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

    return 0
