import math
import time
from typing import Tuple

import cv2
import numpy as np
import onnxruntime

from classes import CLASSES

# 随机颜色
Color_Palette: np.ndarray = np.random.uniform(100, 255, size=(len(CLASSES), 3))


class YOLOV5:
    def __init__(self, model_path: str, confidence_threshold, iou_threshold, providers=None):
        if providers is None:
            providers = ['CPUExecutionProvider']  # cpu

        self.onnx_session = onnxruntime.InferenceSession(model_path, providers=providers)

        # 获取模型指定的信息及输入形状
        self.model_inputs: list = self.onnx_session.get_inputs()
        self.input_shape: list = self.model_inputs[0].shape
        if len(self.input_shape) < 4:
            exit("Model error, please check the model!")
        self.input_width: int = self.input_shape[2]
        self.input_height: int = self.input_shape[3]

        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

    @classmethod
    def preprocess(cls, img: np.ndarray):
        # 将图像颜色空间从BGR转换为RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 通过除以255.0来归一化图像数据
        img_data = np.array(img) / 255.0
        # 转置图像，使通道维度为第一维
        img_data = np.transpose(img_data, (2, 0, 1))  # 通道首
        # 扩展图像数据的维度以匹配预期的输入形状
        img_data = np.expand_dims(img_data, axis=0).astype(np.float32)
        # 返回预处理后的图像数据
        return img_data

    def detect_object_example(self, img: np.ndarray, is_draw: bool = False):
        img, _, (_, _) = self.letterbox(img)
        img_data = self.preprocess(img)
        outputs = self.onnx_session.run(None, {self.model_inputs[0].name: img_data})[0]
        # outputs: (1, 25200, 4+1+n)。25200: 25200个结果; 4: x,y,w,h; 1: 置信度; n: n个类别的得分。
        boxes, classes, scores = self.simple_filter(outputs)
        if boxes is not None:
            if is_draw:
                self.draw(img, boxes, scores, classes)
        return img, boxes, classes, scores

    @classmethod
    def xywh2xyxy(cls, x):
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y

    @classmethod
    def NMS_boxes(cls, boxes, scores):
        """Suppress non-maximal boxes.

        # Arguments
            boxes: ndarray, boxes of objects.
            scores: ndarray, scores of objects.

        # Returns
            keep: ndarray, index of effective boxes.
        """
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        areas = w * h
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])
            w1 = np.maximum(0.0, xx2 - xx1 + 1e-05)
            h1 = np.maximum(0.0, yy2 - yy1 + 1e-05)
            inter = w1 * h1
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            indices = np.where(ovr <= IOU_Threshold)[0]
            order = order[indices + 1]

        keep = np.array(keep)
        return keep

    @classmethod
    def simple_filter(cls, prediction):
        xc = prediction[(Ellipsis, 4)] > Confidence_Threshold
        valid_object = prediction[xc]
        valid_object[:, 5:] *= valid_object[:, 4:5]
        boxes = cls.xywh2xyxy(valid_object[:, :4])
        best_score_class = np.max(valid_object[:, 5:], axis=-1)
        box_classes = np.argmax(valid_object[:, 5:], axis=-1)
        n_boxes, n_classes, n_scores = [], [], []
        for c in set(box_classes):
            indices = np.where(box_classes == c)
            b = boxes[indices]
            c = box_classes[indices]
            s = best_score_class[indices]
            keep = cls.NMS_boxes(b, s)
            if s[keep][0] < Confidence_Threshold:
                print(s[keep])
                continue
            n_boxes.append(b[keep])
            n_classes.append(c[keep])
            n_scores.append(s[keep])

        if not n_classes and not n_scores:
            return None, None, None
        else:
            boxes = np.concatenate(n_boxes)
            classes = np.concatenate(n_classes)
            scores = np.concatenate(n_scores)
            return boxes, classes, scores

    @classmethod
    def draw(cls, img, boxes, scores, classes):
        """Draw the boxes on the image.

        # Argument:
            image: original image.
            boxes: ndarray, boxes of objects.
            classes: ndarray, classes of objects.
            scores: ndarray, scores of objects.
            all_classes: all classes name.
        """
        for box, score, cl in zip(boxes, scores, classes):
            cls.draw_single(img, box, score, cl)

    @classmethod
    def draw_single(cls, img, box, score, cl):
        top, left, right, bottom = box
        print('class: {}, score: {}'.format(CLASSES[cl], score))
        print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))
        top = int(top)
        left = int(left)
        right = int(right)
        bottom = int(bottom)
        cv2.rectangle(img, (top, left), (right, bottom), Color_Palette[cl], 2)
        cv2.putText(img, '{0} {1:.2f}'.format(CLASSES[cl], score), (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    Color_Palette[cl], 2)

    @classmethod
    def letterbox(cls, img: np.ndarray, target_size=(640, 640)) -> Tuple[np.ndarray, float, Tuple[int, int,]]:
        """
        缩放图像，多余部分填充黑边
        :param img: 原图像
        :param target_size: (宽, 高)目标尺寸
        :return: 返回(新图像, 新/原比例, (宽, 高))
        """
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        target_width, target_height = target_size
        height, width = img.shape[:2]

        scale = min(target_width / width, target_height / height)
        resized_width = int(scale * width)
        resized_height = int(scale * height)
        resized_image = cv2.resize(img, (resized_width, resized_height))

        new_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        start_x = (target_width - resized_width) // 2
        start_y = (target_height - resized_height) // 2
        new_image[start_y:start_y + resized_height, start_x:start_x + resized_width] = resized_image

        return new_image, scale, (target_width, target_height)

    @classmethod
    def rotate_image(cls, img, rotate_angle):
        rows, cols = img.shape[:2]
        angle = rotate_angle
        center = (cols / 2, rows / 2)
        new_height = int(cols * abs(math.sin(math.radians(angle))) + rows * abs(math.cos(math.radians(angle))))
        new_width = int(rows * abs(math.sin(math.radians(angle))) + cols * abs(math.cos(math.radians(angle))))
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
        rotation_matrix[(0, 2)] += (new_width - cols) / 2
        rotation_matrix[(1, 2)] += (new_height - rows) / 2
        rotated_image = cv2.warpAffine(img, rotation_matrix, (new_width, new_height))
        return rotated_image


if __name__ == "__main__":
    # 置信度
    Confidence_Threshold: float = 0.5
    # iou阈值
    IOU_Threshold: float = 0.6
    onnx_path = 'runs/coco128_best.onnx'
    my_yolov5 = YOLOV5(onnx_path, Confidence_Threshold, IOU_Threshold)

    original_image = cv2.imread('data/images/bus.jpg')

    t1 = time.time()
    result_img, Boxes, Classes, Scores = my_yolov5.detect_object_example(original_image, True)
    print("time: " + str(time.time() - t1) + "s")
    cv2.imshow('', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.imwrite('res.jpg', result_img)
