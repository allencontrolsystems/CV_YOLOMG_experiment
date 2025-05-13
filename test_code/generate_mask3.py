import cv2
import numpy as np
from FD3_mask import FD3_mask
import os
import time

# all test videos
set0 = ['phantom02', 'phantom03', 'phantom04', 'phantom05', 'phantom08', 'phantom22', 'phantom39',
        'phantom41', 'phantom45', 'phantom47', 'phantom50', 'phantom54', 'phantom55', 'phantom56',
        'phantom57', 'phantom58', 'phantom60', 'phantom61', 'phantom64', 'phantom73', 'phantom79',
        'phantom92', 'phantom93', 'phantom94', 'phantom95', 'phantom97', 'phantom102', 'phantom110',
        'phantom113', 'phantom119', 'phantom133', 'phantom135', 'phantom136', 'phantom141', 'phantom144']

# domain adaptation
set1 = ['phantom02', 'phantom03', 'phantom04', 'phantom05', 'phantom47', 'phantom50',
        'phantom54', 'phantom55', 'phantom56', 'phantom57', 'phantom58', 'phantom60']

# light adaptation
set2 = ['phantom92', 'phantom93', 'phantom94', 'phantom95', 'phantom97', 'phantom133', 'phantom135', 'phantom136']

# test on similar scenes
set3 = ['phantom08', 'phantom22', 'phantom39', 'phantom41', 'phantom45', 'phantom61', 'phantom64',
        'phantom73', 'phantom79', 'phantom102', 'phantom110', 'phantom113', 'phantom119', 'phantom141', 'phantom144']

# train videos
sets = ['phantom09', 'phantom10', 'phantom14', 'phantom17', 'phantom19', 'phantom20', 'phantom28', 'phantom29', 'phantom30', 'phantom32',
        'phantom36', 'phantom40', 'phantom42', 'phantom43', 'phantom44', 'phantom46', 'phantom63', 'phantom65', 'phantom66', 'phantom68',
        'phantom70', 'phantom71', 'phantom74', 'phantom75', 'phantom76', 'phantom77', 'phantom78', 'phantom80', 'phantom81', 'phantom82',
        'phantom84', 'phantom85', 'phantom86', 'phantom87', 'phantom89', 'phantom90', 'phantom101', 'phantom103', 'phantom104', 'phantom105',
        'phantom106', 'phantom107', 'phantom108', 'phantom109', 'phantom111', 'phantom112', 'phantom114', 'phantom115', 'phantom116', 'phantom117',
        'phantom118', 'phantom120', 'phantom132', 'phantom137', 'phantom138', 'phantom139', 'phantom140', 'phantom142', 'phantom143', 'phantom145',
        'phantom146', 'phantom147', 'phantom148', 'phantom149', 'phantom150']


sets_NPS = ['Clip_01', 'Clip_02', 'Clip_03', 'Clip_04', 'Clip_05', 'Clip_06', 'Clip_07', 'Clip_08', 'Clip_09', 'Clip_10',
            'Clip_11', 'Clip_12', 'Clip_13', 'Clip_14', 'Clip_15', 'Clip_16', 'Clip_17', 'Clip_18', 'Clip_19', 'Clip_20',
            'Clip_21', 'Clip_22', 'Clip_23', 'Clip_24', 'Clip_25', 'Clip_26', 'Clip_27', 'Clip_28', 'Clip_29', 'Clip_30',
            'Clip_31', 'Clip_32', 'Clip_33', 'Clip_34', 'Clip_35', 'Clip_36', 'Clip_37', 'Clip_38', 'Clip_39', 'Clip_40',
            'Clip_41', 'Clip_42', 'Clip_43', 'Clip_44', 'Clip_45', 'Clip_46', 'Clip_47', 'Clip_48', 'Clip_49', 'Clip_50']

sets_NPS_test = ['Clip_41', 'Clip_42', 'Clip_43', 'Clip_44', 'Clip_45', 'Clip_46', 'Clip_47', 'Clip_48', 'Clip_49', 'Clip_50']
test_set = ["phantom41(1)", "phantom47", "phantom119", "phantom22"]

t1 = None
t2 = None
videos_len = len(test_set)
for video_count, video_sets in enumerate(test_set):
    video_name = video_sets
    image_save_path = '/home/acs/YOLOMG/demo_results/images/' + video_name
    mask_save_path = '/home/acs/YOLOMG/demo_results/mask31/' + video_name
    video_source_path = '/home/acs/YOLOMG/full_data/ARD100_dataset/test_video/' + video_name + '.mp4'

    if t1 is None:
        took = "N/A"
    else:
        took = t2 - t1

    t1 = time.time()

    # cap = cv2.VideoCapture('/home/acs/YOLOMG/full_data/ARD100_dataset/train_video/' + video_name + '.mp4')
    cap = cv2.VideoCapture(video_source_path)
    lastFrame1 = None
    lastFrame2 = None
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        currentFrame = frame
        count = count + 1
        # save_path = '/home/acs/YOLOMG/full_data/phantom-dataset/images/' + video_name
 

        if not os.path.exists(image_save_path):
            os.makedirs(image_save_path)
        cv2.imwrite(image_save_path + '/' + video_name + '_' + str(count).zfill(4) + '.jpg', currentFrame)

        # frame = cv2.resize(frame, (1280, 1280), dst=None, interpolation=cv2.INTER_CUBIC)

        if lastFrame2 is None:
            if lastFrame1 is None:
                print(' first frame input')
                lastFrame1 = currentFrame
            else:
                print(' second frame input')
                lastFrame2 = currentFrame
            continue

        obj_num = FD3_mask(lastFrame1, lastFrame2, currentFrame, video_name, count-1, mask_save_path)

        print('video name: ', video_name, end=' ')
        print('frame count: %d obj_num: %d' % (count-1, obj_num), f"{videos_len - video_count - 1} more videos to go and last video took {took} s") 
        lastFrame1 = lastFrame2
        lastFrame2 = currentFrame

    cap.release()
    t2 = time.time()


