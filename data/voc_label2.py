# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import os
from os import getcwd

sets = ['train', 'test', 'val']
abs_path = os.getcwd()
print(abs_path)


wd = getcwd()
for image_set in sets:
    image_ids = open('/home/ec2-user/YOLOMG/full_data/ARD100_mask31/ImageSets/Main/%s.txt' % (image_set)).read().strip().split()
    list_file = open('/home/ec2-user/YOLOMG/full_data/ARD100_yolo/%s2.txt' % (image_set), 'w')
    for image_id in image_ids:
        list_file.write('/home/ec2-user/YOLOMG/full_data/ARD100_mask31/mask31/%s.jpg\n' % (image_id))
        #convert_annotation(image_id)
    list_file.close()

