import config as cfg
from PIL import Image as im
import xml.etree.cElementTree as et
import numpy as np
import cv2

def read_img(img_path, size=cfg.image_size):
    #read images and and convert it to square of size 500 by padding it with zeros

    img = cv2.imread(img_path)
    img = np.append(img, np.zeros([500 - img.shape[0], img.shape[1], 3], dtype=np.uint8), axis=0)
    img = np.append(img, np.zeros([500, 500 - img.shape[1], 3], dtype=np.uint8), axis=1)
    small = im.fromarray(img)
    small = small.resize((size, size), im.ANTIALIAS)
    small = np.asarray(small,dtype=np.uint8)

    #return the image after resizing to size
    return small

def center_scale(xmin, ymin, xmax, ymax, img_size=cfg.image_size):
    #convert [xmin, ymin, xmax, ymax] to [centerX, centerY, width, hight] and normalize
    xmin = xmin * img_size / 500
    xmax = xmax * img_size / 500
    ymin = ymin * img_size / 500
    ymax = ymax * img_size / 500

    w = xmax - xmin
    h = ymax - ymin

    cx = (xmax - xmin) / 2
    cy = (ymax - ymin) / 2
    cx += xmin
    cy += ymin
    return (cx, cy, w, h)

def read_label(xml_path, img_size=cfg.image_size, grid_size=cfg.grid_size):
    #reads the labels and makes one label example

    class_to_idx = cfg.classes_map

    classes = cfg.classes_number

    out = np.zeros((grid_size, grid_size, classes + 5),dtype=np.float32)
    r = et.parse(xml_path).getroot()
    for obj in r.findall('object'):
        name = obj.find("name").text
        idx = class_to_idx[name]

        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        x, y, w, h = center_scale(xmin, ymin, xmax, ymax)

        box_size = img_size / grid_size

        cellX = x // box_size
        cellY = y // box_size

        x %= box_size
        y %= box_size

        x /= box_size
        y /= box_size
        w /= img_size
        h /= img_size
        out[cellX][cellY][0] = x
        out[cellX][cellY][1] = y
        out[cellX][cellY][2] = w
        out[cellX][cellY][3] = h
        out[cellX][cellY][4] = 1
        out[cellX][cellY][idx + 4] = 1

    #return one label of shape [grid size, gride size, classes+5]
    return out

