#data and training parameters
image_size=448
max_image_dim=500
classes_number=20
batch_size=32

images_path="C:\\Users\\ahmed\Downloads\\VOCtrainval_11-May-2012\\VOCdevkit\\VOC2012\\JPEGImages"
labels_path="C:\\Users\\ahmed\Downloads\\VOCtrainval_11-May-2012\\VOCdevkit\\VOC2012\\Annotations"

classes_map={"aeroplane": 1,
             "bicycle": 2,
             "bird": 3,
             "boat": 4,
             "bottle": 5,
             "bus": 6,
             'car': 7,
             "cat": 8,
             "chair": 9,
             "cow": 10,
             "diningtable": 11,
             "dog": 12,
             "horse": 13,
             "motorbike": 14,
             "person": 15,
             "pottedplant": 16,
             "sheep": 17,
             "sofa": 18,
             "train": 19,
             "tvmonitor": 20}



#yolo parameters
coord_scale=5
noobj_scale=.5
class_scale=1
obj_scale=1
grid_size=8
boxes_per_cell=2





