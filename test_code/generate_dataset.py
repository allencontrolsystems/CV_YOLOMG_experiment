import os
import shutil
import cv2
import xml.etree.ElementTree as ET
import random
import numpy as np


# ARD100 all videos
set0 = ['phantom02', 'phantom03', 'phantom04', 'phantom05', 'phantom08', 'phantom09', 'phantom10', 'phantom14', 'phantom17', 'phantom19',
        'phantom20', 'phantom22', 'phantom28', 'phantom29', 'phantom30', 'phantom32', 'phantom36', 'phantom39', 'phantom40', 'phantom41',
        'phantom42', 'phantom43', 'phantom44', 'phantom45', 'phantom46', 'phantom47', 'phantom50', 'phantom54', 'phantom55', 'phantom56',
        'phantom57', 'phantom58', 'phantom60', 'phantom61', 'phantom63', 'phantom64', 'phantom65', 'phantom66', 'phantom68', 'phantom70',
        'phantom71', 'phantom73', 'phantom74', 'phantom75', 'phantom76', 'phantom77', 'phantom78', 'phantom79', 'phantom80', 'phantom81',
        'phantom82', 'phantom84', 'phantom85', 'phantom86', 'phantom87', 'phantom89', 'phantom90', 'phantom92', 'phantom93', 'phantom94',
        'phantom95', 'phantom97', 'phantom101', 'phantom102', 'phantom103', 'phantom104', 'phantom105', 'phantom106', 'phantom107', 'phantom108',
        'phantom109', 'phantom110', 'phantom111', 'phantom112', 'phantom113', 'phantom114', 'phantom115', 'phantom116', 'phantom117', 'phantom118',
        'phantom119', 'phantom120', 'phantom132', 'phantom133', 'phantom135', 'phantom136', 'phantom137', 'phantom138', 'phantom139', 'phantom140',
        'phantom141', 'phantom142', 'phantom143', 'phantom144', 'phantom145', 'phantom146', 'phantom147', 'phantom148', 'phantom149', 'phantom150']

# ARD100 train dataset
set1 = ['phantom09', 'phantom10', 'phantom14', 'phantom17', 'phantom19', 'phantom20', 'phantom28', 'phantom29', 'phantom30', 'phantom32',
        'phantom36', 'phantom40', 'phantom42', 'phantom43', 'phantom44', 'phantom46', 'phantom63', 'phantom65', 'phantom66', 'phantom68',
        'phantom70', 'phantom71', 'phantom74', 'phantom75', 'phantom76', 'phantom77', 'phantom78', 'phantom80', 'phantom81', 'phantom82',
        'phantom84', 'phantom85', 'phantom86', 'phantom87', 'phantom89', 'phantom90', 'phantom101', 'phantom103', 'phantom104', 'phantom105',
        'phantom106', 'phantom107', 'phantom108', 'phantom109', 'phantom111', 'phantom112', 'phantom114', 'phantom115', 'phantom116', 'phantom117',
        'phantom118', 'phantom120', 'phantom132', 'phantom137', 'phantom138', 'phantom139', 'phantom140', 'phantom142', 'phantom143', 'phantom145',
        'phantom146', 'phantom147', 'phantom148', 'phantom149', 'phantom150']

# NPS train dataset
sets_NPS = ['Clip_01', 'Clip_02', 'Clip_03', 'Clip_04', 'Clip_05', 'Clip_06', 'Clip_07', 'Clip_08', 'Clip_09', 'Clip_10',
            'Clip_11', 'Clip_12', 'Clip_13', 'Clip_14', 'Clip_15', 'Clip_16', 'Clip_17', 'Clip_18', 'Clip_19', 'Clip_20',
            'Clip_21', 'Clip_22', 'Clip_23', 'Clip_24', 'Clip_25', 'Clip_26', 'Clip_27', 'Clip_28', 'Clip_29', 'Clip_30',
            'Clip_31', 'Clip_32', 'Clip_33', 'Clip_34', 'Clip_35', 'Clip_36', 'Clip_37', 'Clip_38', 'Clip_39', 'Clip_40']

# NPS test dataset
sets_NPS_test = ['Clip_41', 'Clip_42', 'Clip_43', 'Clip_44', 'Clip_45', 'Clip_46', 'Clip_47', 'Clip_48', 'Clip_49', 'Clip_50']

# ARD100 test videos
set2 = ['phantom02', 'phantom03', 'phantom04', 'phantom05', 'phantom08', 'phantom22', 'phantom39',
        'phantom41', 'phantom45', 'phantom47', 'phantom50', 'phantom54', 'phantom55', 'phantom56',
        'phantom57', 'phantom58', 'phantom60', 'phantom61', 'phantom64', 'phantom73', 'phantom79',
        'phantom92', 'phantom93', 'phantom94', 'phantom95', 'phantom97', 'phantom102', 'phantom110',
        'phantom113', 'phantom119', 'phantom133', 'phantom135', 'phantom136', 'phantom141', 'phantom144']

# domain adaptation
# new scenes
sets_new_scenes = ['phantom02', 'phantom03', 'phantom04', 'phantom05', 'phantom47', 'phantom50',
                   'phantom54', 'phantom55', 'phantom56', 'phantom57', 'phantom58', 'phantom60']

# low light adaptation
sets_low = ['phantom95', 'phantom97', 'phantom133', 'phantom135', 'phantom136']

small_num = 0

# different size test
set_es = ['phantom04', 'phantom22', 'phantom39', 'phantom41', 'phantom45', 'phantom50', 'phantom54', 'phantom55', 'phantom61', 'phantom64', 'phantom73', 'phantom94']  # smaller than 144
set_rs = ['phantom02', 'phantom56', 'phantom57', 'phantom58', 'phantom60', 'phantom79', 'phantom92', 'phantom102', 'phantom110', 'phantom113', 'phantom119', 'phantom141', 'phantom144']  # 144~400
set_gs = ['phantom03', 'phantom05', 'phantom47', 'phantom93']  # 400~1024

video_lens = len(set1)
for video_count, video_sets in enumerate(set1):
    id = video_sets
    imgdir = "/home/ec2-user/YOLOMG/full_data/phantom-dataset/images/" + id + "/"
    annodir = '/home/ec2-user/YOLOMG/full_data/phantom-dataset/annotations/' + id + '/'
    maskdir = '/home/ec2-user/YOLOMG/full_data/phantom-dataset/mask31/' + id + '/'

    imgdest = '/home/ec2-user/YOLOMG/full_data/ARD100_mask31/images/'
    annodest = '/home/ec2-user/YOLOMG/full_data/ARD100_mask31/annotations/'
    maskdest = '/home/ec2-user/YOLOMG/full_data/ARD100_mask31/mask31/'

    if not os.path.exists(imgdest):
        os.makedirs(imgdest)

    if not os.path.exists(annodest):
        os.makedirs(annodest)

    if not os.path.exists(maskdest):
        os.makedirs(maskdest)

    # end_index = int(len(image_list)/3)

    # for image in image_list:
    #     img = cv2.imread(imgdir + image)
    #     img_prefix = image.split('_')[0]
    #     # pic_height, pic_width, pic_depth = img.shape[0], img.shape[1], img.shape[2]
    #     break

    image_list = os.listdir(maskdir)
    num_of_image = len(image_list)
    end_index = int(num_of_image/1)
    # num_of_train = int(num_of_image * 0.4)
    # train_list = np.sort(random.sample(image_list, num_of_train))

    # for image in train_list:
    for i in range(end_index):
        image = id + '_' + str(i*1 + 2).zfill(4)
        name = image.split(".")
        imgname = name[0] + '.jpg'
        xmlname = name[0] + '.xml'

        img_path = os.path.join(imgdir, imgname)
        mask_path = os.path.join(maskdir, imgname)
        xml_path = os.path.join(annodir, xmlname)

        if not os.path.exists(xml_path):
            continue

        tree = ET.parse(xml_path)
        root = tree.getroot()

        if root.find('object') is None:
            continue

        for obj in root.iter('object'):
            xmlbox = obj.find('bndbox')
            b1 = float(xmlbox.find('xmin').text)
            b2 = float(xmlbox.find('xmax').text)
            b3 = float(xmlbox.find('ymin').text)
            b4 = float(xmlbox.find('ymax').text)
            area = (b2 - b1) * (b4 - b3)

        # 对于global detector, area_thresh = 12 * 12, 对于local detector, area_thresh = 25
        if area >= 25:
            shutil.copy(img_path, imgdest)
            shutil.copy(mask_path, maskdest)
            shutil.copy(xml_path, annodest)
        else:
            # print(xmlname, end=' ')
            small_num = small_num + 1
            # print('too small object: ', area)

    print('small object num: ', small_num, " remaining videos: ", video_lens - video_count)

# for i in range(end_index):
#     # for image in image_list:
#     num = str(i * 3 + 1)
#     # num = str(i)
#     imgname = id + '_' + num.zfill(4) + '.jpg'
#     xmlname = id + '_' + num.zfill(4) + '.xml'
#
#     # image_pre, ext = os.path.splitext(image)
#     # imgname = image_pre + '.jpg'
#     # xmlname = image_pre + '.xml'
#
#     img_path = os.path.join(imgdir, imgname)
#     xml_path = os.path.join(annodir, xmlname)
#
#     if not os.path.exists(xml_path):
#         continue
#
#     tree = ET.parse(xml_path)
#     root = tree.getroot()
#
#     if root.find('object') is None:
#         continue
#
#     # if not os.path.exists(img_path):
#     #     continue
#
#     shutil.copy(img_path, imgdest)
#     shutil.copy(xml_path, annodest)
#     print(xmlname)
