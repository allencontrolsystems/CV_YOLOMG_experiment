# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import os
from os import getcwd

sets = ['train', 'test', 'val']
abs_path = os.getcwd()
print(abs_path)


wd = getcwd()
for image_set in sets:
    image_ids = open('/home/xxx/Documents/YOLOMGt/dataset/NPS3/ImageSets/Main/%s.txt' % (image_set)).read().strip().split()
    list_file = open('/home/xxx/Documents/YOLOMG/dataset/NPS3/%s.txt' % (image_set), 'w')
    for image_id in image_ids:
        list_file.write('/home/xxx/Documents/YOLOMG/dataset/NPS3/images/%s.jpg\n' % (image_id))
        #convert_annotation(image_id)
    list_file.close()

