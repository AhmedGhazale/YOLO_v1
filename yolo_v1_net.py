import tensorflow as tf
import config as cfg


class YOLO:
    def __init__(self, is_training=True):

        self.images = tf.placeholder(tf.float32, [None, cfg.image_size, cfg.image_size, 3], name="images")
        self.labels = tf.placeholder(tf.float32, [None, cfg.grid_size, cfg.grid_size, cfg.classes_number + 5])
        self.drop_out = cfg.drop_out_rate
        self.learning_rate = cfg.learning_rate
        self.is_training = is_training
        self.grid_size = cfg.grid_size
        self.boxes_per_cell = cfg.boxes_per_cell
        self.classes_number = cfg.classes_number
        self.output_size = self.grid_size * self.grid_size * (self.boxes_per_cell * 5 + self.classes_number)
        self.logits = self.build_model(images=self.images,
                                       rate=self.drop_out,
                                       is_training=self.is_training,
                                       outputs_num=self.output_size)



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

        model = tf.layers.dense(model, outputs_num, name="logits")

        return model

    def calc_iou(self, boxes1, boxes2):
        # TODO
        return

    def loss(self):
        # TODO
        return


    def get_pridictios(self, images):
        # TODO
        return
