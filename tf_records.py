import sys
from utils import read_img,read_label
import config as cfg


import tensorflow as tf



def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_data_record(out_filename, images_addrs, labels_addrs):
    # open the TFRecords file
    writer = tf.python_io.TFRecordWriter(out_filename)
    for i in range(len(images_addrs)):
        # print how many images are saved every 100 images
        if  i % 100==0:
            print('Train data: {}/{}'.format(i, len(images_addrs)))
            sys.stdout.flush()
        # Load the image
        img = read_img(images_addrs[i])

        label = read_label(labels_addrs[i])
        if img  is None or label is None:
            print("erroe null image")
            continue

        # Create a feature
        feature = {
            'image_raw': _bytes_feature(img.tostring()),
            'label': _bytes_feature(label.tostring())
        }
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()

def create_tfrecords(name):

    names=[]#names of the data example
    for line in open("names.txt",'r'):
        names.append(line[0:-1])

        images_addresses=[]
        labels_addresses=[]

        for name in names:
            images_addresses.append(cfg.images_path+"\\"+name+".jpg")
            labels_addresses.append(cfg.labels_path + "\\" + name + ".xml")

        create_data_record(name+'.tfrecords', images_addresses, labels_addresses)


def parser(record):
    #used to map parse the encoded examples back to image and label
    keys_to_features = {
        "image_raw": tf.FixedLenFeature([], tf.string),
        "label":     tf.FixedLenFeature([], tf.string)
    }
    parsed = tf.parse_single_example(record, keys_to_features)
    image = tf.decode_raw(parsed["image_raw"], tf.uint8)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, shape=[cfg.image_size, cfg.image_size, 3])

    label = tf.decode_raw(parsed["label"], tf.float32)

    label = tf.reshape(label, shape=[cfg.grid_size, cfg.grid_size, cfg.classes_number+5])

    return  image, label

def read_dataset(dataset_name, batch_size=cfg.batch_size, repeat=None, shuflle=True):
    dataset = tf.data.TFRecordDataset(filenames=[dataset_name], num_parallel_reads=batch_size)
    dataset = dataset.repeat(repeat).map(parser).shuffle(shuflle).prefetch(2)

    #return the dataset in a tf.data.Dataset object
    return dataset
