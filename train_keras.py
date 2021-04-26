#!/usr/bin/python3

from os import environ, listdir;
from os.path import join, exists;
from math import ceil;
import numpy as np;
import cv2;
import tensorflow as tf;
from models import YOLOv4, Loss;
from Predictor import Predictor;
from create_dataset import parse_function_generator, parse_function;

environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1';
#environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices --tf_xla_auto_jit=2'
#os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3';
#os.environ['CUDA_VISIBLE_DEVICES'] = '';
batch_size = 4; # images of different sizes can't be stack into a batch
trainset_size = 118287;
testset_size = 5000;

def main():

  yolov4 = YOLOv4((608,608,3,), 80);
  loss1 = Loss((608,608,3,), 0, 80);
  loss2 = Loss((608,608,3,), 1, 80);
  loss3 = Loss((608,608,3,), 2, 80);
  if exists('./checkpoints/ckpt'): yolov4.load_weights('./checkpoints/ckpt/variables/variables');
  optimizer = tf.keras.optimizers.Adam(1e-4);
  yolov4.compile(optimizer = optimizer, loss = {'output1': lambda labels, outputs: loss1([outputs, labels]),
                                                                     'output2': lambda labels, outputs: loss2([outputs, labels]),
                                                                     'output3': lambda labels, outputs: loss3([outputs, labels])});

  # load downloaded dataset
  trainset_filenames = [join('trainset', filename) for filename in listdir('trainset')];
  testset_filenames = [join('testset', filename) for filename in listdir('testset')];
  trainset = tf.data.TFRecordDataset(trainset_filenames).map(parse_function_generator(80)).shuffle(batch_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE);
  testset = tf.data.TFRecordDataset(testset_filenames).map(parse_function_generator(80)).shuffle(batch_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE);
  callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir = './checkpoints'),
    tf.keras.callbacks.ModelCheckpoint(filepath = './checkpoints/ckpt', save_freq = 10000),
  ];
  yolov4.fit(trainset, steps_per_epoch = ceil(trainset_size / batch_size), epochs = 100, validation_data = testset, validation_steps = ceil(testset_size / batch_size), callbacks = callbacks);
  yolov4.save('yolov4.h5');

if __name__ == "__main__":
  
  main();

