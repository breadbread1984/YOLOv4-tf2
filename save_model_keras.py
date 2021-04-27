#!/usr/bin/python3

from shutil import rmtree;
from os import mkdir;
from os.path import exists;
import tensorflow as tf;
from models import YOLOv4;

def main():

  yolov4 = YOLOv4((608, 608, 3), 80);
  yolov4.load_weights('./checkpoints/ckpt');
  if exists('trained_model'): rmtree('trained_model');
  mkdir('trained_model');
  yolov4.save_weights('trained_model/yolov4', save_format = 'tf');

if __name__ == "__main__":

  main();

