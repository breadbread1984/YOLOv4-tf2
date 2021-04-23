#!/usr/bin/python3

from absl import app, flags;
from os.path import join;
from pycocotools.coco import COCO;
from pycocotools.cocoeval import COCOeval;
import numpy as np;
import cv2;
import tensorflow as tf;
from Predictor import Predictor;

FLAGS = flags.FLAGS;
flags.DEFINE_string('model', 'yolov3.h5', 'path to model file to evaluate');
flags.DEFINE_string('coco_eval_dir', None, 'path to coco evaluate directory');
flags.DEFINE_string('annotation_dir', None, 'path to annotation directory');

debug_mode = False;

label_map = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, -1, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, -1, 25, 26, -1, -1, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, -1, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, -1, 61, -1, -1, 62, -1, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, -1, 74, 75, 76, 77, 78, 79, 80];

def main(argv):

  yolov4 = tf.keras.models.load_model(FLAGS.model, compile = False);
  predictor = Predictor(yolov4 = yolov4);
  anno = COCO(join(FLAGS.annotation_dir, 'instances_val2017.json'));
  count = 1;
  for imgid in anno.getImgIds():
    print("processing (%d/%d)" % (count, len(anno.getImgIds())));
    detections = list();
    # predict
    img_info = anno.loadImgs([imgid])[0];
    img = cv2.imread(join(FLAGS.coco_eval_dir, img_info['file_name']));
    boundings = predictor.predict(img).numpy();
    # collect results
    if debug_mode:
      color_map = dict();
      img_gt = img.copy();
    for bounding in boundings:
      detections.append([imgid, bounding[0], bounding[1], bounding[2] - bounding[0], bounding[3] - bounding[1], bounding[4], label_map.index(int(bounding[5]) + 1)]);
      if debug_mode:
        if bounding[5].astype('int32') not in color_map:
          color_map[bounding[5].astype('int32')] = tuple(np.random.randint(low = 0, high = 256, size = (3,)).tolist());
        cv2.rectangle(img, tuple(bounding[0:2].astype('int32').tolist()), tuple(bounding[2:4].astype('int32').tolist()), color_map[bounding[5].astype('int32')], 1);
        cv2.putText(img, list(filter(lambda x: x['id'] == label_map.index(int(bounding[5]) + 1), anno.dataset['categories']))[0]['name'], tuple(bounding[0:2].astype('int32').tolist()), cv2.FONT_HERSHEY_PLAIN, 1, color_map[bounding[5].astype('int32')], 2);
    if debug_mode:
      annIds = anno.getAnnIds(imgIds = imgid);
      anns = anno.loadAnns(annIds);
      for ann in anns:
        bbox_x, bbox_y, bbox_w, bbox_h = ann['bbox'];
        if label_map[ann['category_id']] - 1 not in color_map:
          color_map[label_map[ann['category_id']] - 1] = tuple(np.random.randint(low = 0, high = 256, size = (3,)).tolist());
        cv2.rectangle(img_gt, (int(bbox_x), int(bbox_y), int(bbox_w), int(bbox_h)), color_map[label_map[ann['category_id']] - 1], 1);
        cv2.putText(img_gt, list(filter(lambda x: x['id'] == ann['category_id'], anno.dataset['categories']))[0]['name'], (int(bbox_x), int(bbox_y)), cv2.FONT_HERSHEY_PLAIN, 1, color_map[label_map[ann['category_id']] - 1], 2);
    if debug_mode:
      stacked = np.concatenate([img, img_gt], axis = 0)
      cv2.imshow('detect (up), ground truth (down)', stacked);
      cv2.waitKey();
    count += 1;
  cocoDt = anno.loadRes(np.array(detections));
  cocoEval = COCOeval(anno, cocoDt, iouType = 'bbox');
  cocoEval.params.imgIds = anno.getImgIds();
  cocoEval.evaluate();
  cocoEval.accumulate();
  cocoEval.summarize();

if __name__ == "__main__":

  app.run(main);

