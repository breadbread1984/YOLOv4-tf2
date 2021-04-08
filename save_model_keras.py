#!/usr/bin/python3

import tensorflow as tf;
from models import YOLOv4;

def main():

  yolov4 = YOLOv4((608, 608, 3), 80);
  yolov4.load_weights('./checkpoints/ckpt/variables/variables');
  yolov4.save('yolov4.h5');
  yolov4.save_weights('yolov4_weights.h5');

if __name__ == "__main__":

  assert tf.executing_eagerly();
  main();

