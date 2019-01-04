#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: 1.imageToChar.py
# @time: 2018/12/12 下午2:31
# @desc:

import numpy as np
from PIL import Image
import time


if __name__ == '__main__':
    image_file = 'timg.jpeg'
    # image_file = 'WechatIMG36.jpeg'
    height = 80

    img = Image.open(image_file)
    img_width, img_height = img.size
    width = 2 * height * img_width // img_height  # 假定字符的高度是宽度的2倍
    img = img.resize((width, height), Image.ANTIALIAS)
    pixels = np.array(img.convert('L'))
    chars = "MNHQ$OC?7>!:-;. "
    N = len(chars)
    step = 256 // N
    for i in range(height):
        result = ''
        for j in range(width):
            result += chars[pixels[i][j] // step]
        # result += '\n'
        print result
        time.sleep(0.5)
    # with open('image.txt', mode='w') as f:
    #     f.write(result)


