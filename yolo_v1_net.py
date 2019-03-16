import tensorflow as tf
import numpy as np
import config as cfg


class YOLO:
    def __init__(self, is_training=True):

        self.images = tf.placeholder(tf.float32, [None, cfg.image_size, cfg.image_size, 3], name="images")
        self.labels = tf.placeholder(tf.float32, [None, cfg.grid_size, cfg.grid_size, cfg.classes_number + 5])
        self.batch_size = cfg.batch_size
        self.drop_out = cfg.drop_out_rate
        self.learning_rate = cfg.learning_rate
        self.coord_scale = cfg.coord_scale
        self.object_scale = cfg.obj_scale
        self.no_object_scale = cfg.noobj_scale
        self.class_scale = cfg.class_scale

        self.coord_scale = cfg.coord_scale
        self.coord_scale = cfg.coord_scale

        self.is_training = is_training
        self.grid_size = cfg.grid_size
        self.boxes_per_cell = cfg.boxes_per_cell
        self.classes_number = cfg.classes_number
        self.output_size = self.grid_size * self.grid_size * (self.boxes_per_cell * 5 + self.classes_number)
        self.offset = np.transpose(np.reshape(np.array(
            [np.arange(self.grid_size)] * self.grid_size * self.boxes_per_cell),
            (self.boxes_per_cell, self.grid_size, self.grid_size)), (1, 2, 0))
        self.classes_outs = self.grid_size * self.grid_size * self.classes_number
        self.boxes_outs = self.grid_size * self.grid_size * self.boxes_per_cell * 4
        self.score_outs = self.grid_size * self.grid_size * self.boxes_per_cell

        self.boundry1=self.classes_outs
        self.boundry2 = self.boundry1 + self.score_outs
        self.boundry3 = self.boundry2 + self.boxes_outs

        self.logits = self.build_model(images=self.images,
                                       rate=self.drop_out,
                                       is_training=self.is_training,
                                       outputs_num=self.output_size)
        self.labels = tf.placeholder(tf.float32,[None,self.grid_size,self.grid_size,self.classes_number+5],name="labels")
        self.total_loss= self.loss(logits=self.logits,outs = self.labels)

    def build_model(self, images, rate, is_training, outputs_num):
        images = tf.div(images, 255.0, name="scale")
        model = tf.nn.leaky_relu(
            tf.layers.batch_normalization(tf.layers.conv2d(images, 64, 7, padding="SAME", strides=(2, 2))),
            name="conv1")
        model = tf.nn.leaky_relu(tf.layers.batch_normalization(tf.layers.conv2d(model, 128, 3, padding="SAME")),
                                 name="conv2")

        model = tf.layers.max_pooling2d(model, 2, 2, name="pool1")

        model = tf.nn.leaky_relu(tf.layers.batch_normalization(tf.layers.conv2d(model, 128, 1, padding="SAME")),
                                 name="conv3")
        model = tf.nn.leaky_relu(tf.layers.batch_normalization(tf.layers.conv2d(model, 256, 3, padding="SAME")),
                                 name="conv4")
        model = tf.nn.leaky_relu(tf.layers.batch_normalization(tf.layers.conv2d(model, 256, 1, padding="SAME")),
                                 name="conv5")

        model = tf.layers.max_pooling2d(model, 2, 2, name="pool2")

        model = tf.nn.leaky_relu(tf.layers.batch_normalization(tf.layers.conv2d(model, 512, 3, padding="SAME")),
                                 name="conv6")
        model = tf.nn.leaky_relu(tf.layers.batch_normalization(tf.layers.conv2d(model, 256, 1, padding="SAME")),
                                 name="conv7")
        model = tf.nn.leaky_relu(tf.layers.batch_normalization(tf.layers.conv2d(model, 512, 3, padding="SAME")),
                                 name="conv8")
        model = tf.nn.leaky_relu(tf.layers.batch_normalization(tf.layers.conv2d(model, 512, 1, padding="SAME")),
                                 name="conv9")
        model = tf.nn.leaky_relu(tf.layers.batch_normalization(tf.layers.conv2d(model, 1024, 3, padding="SAME")),
                                 name="conv10")

        model = tf.layers.max_pooling2d(model, 2, 2, name="pool3")

        model = tf.nn.leaky_relu(tf.layers.batch_normalization(tf.layers.conv2d(model, 512, 1, padding="SAME")),
                                 name="conv11")
        model = tf.nn.leaky_relu(tf.layers.batch_normalization(tf.layers.conv2d(model, 1024, 3, padding="SAME")),
                                 name="conv12")
        model = tf.nn.leaky_relu(tf.layers.batch_normalization(tf.layers.conv2d(model, 512, 1, padding="SAME")),
                                 name="conv13")
        model = tf.nn.leaky_relu( tf.layers.batch_normalization(tf.layers.conv2d(model, 1024, 3, padding="SAME", strides=(2, 2))),
                                 name="conv14")
        model = tf.nn.leaky_relu(tf.layers.batch_normalization(tf.layers.conv2d(model, 1024, 3, padding="SAME")),
                                 name="conv15")

        model = tf.layers.flatten(model, name='flat')

        model = tf.nn.leaky_relu(tf.layers.batch_normalization(tf.layers.dense(model, 4096)), name="fully_connected1")
        model = tf.layers.dropout(model, rate=rate, training=is_training, name="dropout")

        model = tf.layers.dense(model, outputs_num)
        model = tf.add(model,0.0,name="logits")

        return model

    def calc_iou(self, boxes1, boxes2):
        # converts [x, y, w, h] to [xmin, ymin, xmax, ymax]
        boxes1=tf.stack([
            boxes1[..., 0] - boxes1[..., 2]/2.0,
            boxes1[..., 1] - boxes1[..., 3] / 2.0,
            boxes1[..., 0] + boxes1[..., 2] / 2.0,
            boxes1[..., 1] + boxes1[..., 3] / 2.0
        ],axis=-1)
        boxes2 = tf.stack([
            boxes2[..., 0] - boxes2[..., 2] / 2.0,
            boxes2[..., 1] - boxes2[..., 3] / 2.0,
            boxes2[..., 0] + boxes2[..., 2] / 2.0,
            boxes2[..., 1] + boxes2[..., 3] / 2.0
        ], axis=-1)

        tl=tf.maximum(boxes1[...,:2],boxes2[...,:2])
        br=tf.minimum(boxes1[...,2:],boxes2[...,2:])

        intersection=tf.maximum(0.0,br-tl)

        intersection_area=intersection[...,0] * intersection[...,1]

        boxes1_area=boxes1[...,2]*boxes1[...,3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        union_area=tf.maximum (boxes1_area + boxes2_area - intersection_area,1e-9)

        return tf.clip_by_value(intersection_area / union_area,0.0,1.0)

    def loss(self,logits , outs):
        # divide the logits to the different terms
        prid_classes = tf.reshape( logits[:,:self.boundry1],[-1,self.grid_size,self.grid_size,self.classes_number])
        prid_scores = tf.reshape(logits[:,self.boundry1: self.boundry2],[-1,self.grid_size,self.grid_size,self.boxes_per_cell])
        prid_boxes = tf.reshape(logits[:,self.boundry2:self.boundry3],[-1,self.grid_size, self.grid_size,self.boxes_per_cell,4])

        # divide the labels to the different terms
        out_boxes = tf.reshape (outs[...,0:4],[-1,self.grid_size,self.grid_size,1,4])
        out_boxes = tf.tile(out_boxes,[1,1,1,2,1])
        out_scores = outs[...,4:5]
        out_classes = outs[...,5:]

        # offset is used to make the x,y coord normalized over the hole images and not the cell only
        offset = tf.expand_dims(tf.constant(self.offset,dtype=tf.float32),0)
        offset_X = tf.tile(offset ,[self.batch_size,1,1,1])
        offset_Y = tf.transpose(offset_X,[0, 2, 1, 3])

        # convert the boxes to be usable with the iou function
        prid_boxes_iou = tf.stack([
            (prid_boxes[..., 0]+offset_X)/self.grid_size,
            (prid_boxes[..., 1] + offset_Y) / self.grid_size,
            tf.square(prid_boxes[...,2]),
            tf.square(prid_boxes[..., 3])
        ],axis=-1)

        out_boxes_iou = tf.stack([
            (out_boxes[..., 0] + offset_X) / self.grid_size,
            (out_boxes[..., 1] + offset_Y) / self.grid_size,
            out_boxes[..., 2],
            out_boxes[..., 3]
        ], axis=-1)

        # calc the iou
        iou = self.calc_iou(prid_boxes_iou,out_boxes_iou)
        iou_max = tf.reduce_max(iou,-1,keepdims=True)

        # make the object mask and no object mask
        object_mask = tf.cast((iou>=iou_max),tf.float32) * out_scores
        no_object_mask = tf.ones_like(object_mask,tf.float32)-object_mask

        # classes_loss
        class_loss = out_scores * tf.reduce_sum(tf.square(out_classes-prid_classes),axis=-1,keepdims=True)
        class_loss = tf.reduce_mean( tf.reduce_sum(class_loss,axis=[1,2,3])) * self.class_scale

        # object loss
        object_loss = object_mask * tf.square( iou - prid_scores )
        object_loss = tf.reduce_mean( tf.reduce_sum(object_loss ,axis=[1, 2, 3])) * self.object_scale

        # no object loss
        no_object_loss = tf.square(no_object_mask * prid_scores)
        no_object_loss = tf.reduce_mean(tf.reduce_sum(no_object_loss, axis=[1, 2, 3])) * self.no_object_scale


        # convert the boxes to be in used in loss
        out_boxes_loss = tf.stack([
            out_boxes[..., 0],
            out_boxes[..., 1],
            tf.sqrt(out_boxes[..., 2]),
            tf.sqrt(out_boxes[..., 3])
        ], axis=-1)

        # booxes loss
        coord_loss = object_mask * tf.reduce_sum(tf.square( out_boxes_loss - prid_boxes ),axis=-1)

        coord_loss = tf.reduce_mean( tf.reduce_sum(coord_loss,axis=[1,2,3] ))  * self.coord_scale


        return tf.add(class_loss + no_object_loss + object_loss , coord_loss,name="loss")





    def get_pridictios(self, images):
        # TODO
        return
