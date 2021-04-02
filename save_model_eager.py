#!/usr/bin/python3

import tensorflow as tf;
from models import YOLOv4;

def main():

  yolov4 = YOLOv4((416, 416, 3), 80);
  optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.ExponentialDecay(1e-4, decay_steps = 110000, decay_rate = 0.99));
  checkpoint = tf.train.Checkpoint(model = yolov4, optimizer = optimizer);
  checkpoint.restore(tf.train.latest_checkpoint('checkpoints'));
  yolov4.save('yolov4.h5');
  yolov4.save_weights('yolov4_weights.h5');

if __name__ == "__main__":

  assert tf.executing_eagerly();
  main();

