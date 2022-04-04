import sys
sys.path.append("../../../../common")
sys.path.append("../")
import os
import numpy as np
import acl
import cv2 as cv
from PIL import Image
import csv
import constants as const
from acllite_model import AclLiteModel
from acllite_resource import AclLiteResource

labels = ["face", "person", "mask"]
INPUT_DIR_1 = '../HV_MASK/'
INPUT_DIR_2 = '../NO_MASK/'
OUTPUT_DIR = '../out/'
MODEL_PATH = "../model/mask_detection.om"
MODEL_WIDTH = 640
MODEL_HEIGHT = 352
class_num = 3
stride_list = [8, 16, 32]
anchors_1 = np.array([[10, 13], [16, 30], [33, 23]]) / stride_list[0]
anchors_2 = np.array([[30, 61], [62, 45], [59, 119]]) / stride_list[1]
anchors_3 = np.array([[116, 90], [156, 198], [163, 326]]) / stride_list[2]
anchor_list = [anchors_1, anchors_2, anchors_3]

conf_threshold = 0.2
iou_threshold = 0.3

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0)]

def preprocess(img_path):
    image = Image.open(img_path)
    img_h = image.size[1]
    img_w = image.size[0]
    net_h = MODEL_HEIGHT
    net_w = MODEL_WIDTH

    scale = min(float(net_w) / float(img_w), float(net_h) / float(img_h))
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)

    shift_x = (net_w - new_w) // 2
    shift_y = (net_h - new_h) // 2
    shift_x_ratio = (net_w - new_w) / 2.0 / net_w
    shift_y_ratio = (net_h - new_h) / 2.0 / net_h

    image_ = image.resize( (new_w, new_h))
    new_image = np.zeros((net_h, net_w, 3), np.uint8)
    new_image[shift_y: new_h + shift_y, shift_x: new_w + shift_x, :] = np.array(image_)
    new_image = new_image.astype(np.float32)
    new_image = new_image / 255

    return new_image, image

def overlap(x1, x2, x3, x4):
    left = max(x1, x3)
    right = min(x2, x4)
    return right - left

def cal_iou(box, truth):
    #这个就是计算IOU的
    w = overlap(box[0], box[2], truth[0], truth[2])
    h = overlap(box[1], box[3], truth[1], truth[3])
    #为什么这里都只有一个值？坐标不应该是两个值吗？
    if w <= 0 or h <= 0:
        return 0
    inter_area = w * h
    union_area = (box[2] - box[0]) * (box[3] - box[1]) + (truth[2] - truth[0]) * (truth[3] - truth[1]) - inter_area
    return inter_area * 1.0 / union_area

def apply_nms(all_boxes, thres):
    res = []
    for cls in range(class_num):
        cls_bboxes = all_boxes[cls]
        sorted_boxes = sorted(cls_bboxes, key=lambda d: d[5])[::-1]
        #看起来是按照第五个维度上的数值进行了逆序排列（第五个维度是？好像是把置信度（评分）最高的图片作为真实值，用这个来减少检测框的个数？
        p = dict()
        for i in range(len(sorted_boxes)):
            if i in p:
                continue
            truth = sorted_boxes[i]
            #################################
            #重要，这个truth是如何计算出来的？为什么对cls_bboxes做了排列？这个为什么就是真实值了？
            #测试用例是有标注的吗？
            #################################
            for j in range(i + 1, len(sorted_boxes)):
                if j in p:
                    continue
                box = sorted_boxes[j]
                iou = cal_iou(box, truth)
                if iou >= thres:
                    p[j] = 1
        for i in range(len(sorted_boxes)):
            if i not in p:
                res.append(sorted_boxes[i])
    return res

def decode_bbox(conv_output, anchors, img_w, img_h, x_scale, y_scale, shift_x_ratio, shift_y_ratio):
    def _sigmoid(x):
        s = 1 / (1 + np.exp(-x))
        return s

    h, w, _ = conv_output.shape
    pred = conv_output.reshape((h * w, 3, 5 + class_num))#这个东西输出看一下是什么
    #主要是怎么做实际的标注？
    pred[..., 4:] = _sigmoid(pred[..., 4:])
    pred[..., 0] = (_sigmoid(pred[..., 0]) + np.tile(range(w), (3, h)).transpose((1, 0))) / w
    pred[..., 1] = (_sigmoid(pred[..., 1]) + np.tile(np.repeat(range(h), w), (3, 1)).transpose((1, 0))) / h
    pred[..., 2] = np.exp(pred[..., 2]) * anchors[:, 0:1].transpose((1, 0)) / w
    pred[..., 3] = np.exp(pred[..., 3]) * anchors[:, 1:2].transpose((1, 0)) / h

    bbox = np.zeros((h * w, 3, 4))
    bbox[..., 0] = np.maximum((pred[..., 0] - pred[..., 2] / 2.0 - shift_x_ratio) * x_scale * img_w, 0)  # x_min
    bbox[..., 1] = np.maximum((pred[..., 1] - pred[..., 3] / 2.0 - shift_y_ratio) * y_scale * img_h, 0)  # y_min
    bbox[..., 2] = np.minimum((pred[..., 0] + pred[..., 2] / 2.0 - shift_x_ratio) * x_scale * img_w, img_w)  # x_max
    bbox[..., 3] = np.minimum((pred[..., 1] + pred[..., 3] / 2.0 - shift_y_ratio) * y_scale * img_h, img_h)  # y_max
    #这里得到的bbox是个张量
    pred[..., :4] = bbox
    pred = pred.reshape((-1, 5 + class_num))
    pred[:, 4] = pred[:, 4] * pred[:, 5:].max(1)
    pred = pred[pred[:, 4] >= conf_threshold]
    pred[:, 5] = np.argmax(pred[:, 5:], axis=-1)

    all_boxes = [[] for ix in range(class_num)]
    for ix in range(pred.shape[0]):
        box = [int(pred[ix, iy]) for iy in range(4)]
        box.append(int(pred[ix, 5]))
        box.append(pred[ix, 4])
        all_boxes[box[4] - 1].append(box)
    
    return all_boxes

def convert_labels(label_list):
    if isinstance(label_list, np.ndarray):
        label_list = label_list.tolist()
        label_names = [labels[int(index)] for index in label_list]
    return label_names

def post_process(infer_output, origin_img):
    print("post process")
    result_return = dict()
    img_h = origin_img.size[1]
    img_w = origin_img.size[0]
    scale = min(float(MODEL_WIDTH) / float(img_w), float(MODEL_HEIGHT) / float(img_h))
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)
    
    shift_x_ratio = (MODEL_WIDTH - new_w) / 2.0 / MODEL_WIDTH
    shift_y_ratio = (MODEL_HEIGHT - new_h) / 2.0 / MODEL_HEIGHT
    class_number = len(labels)
    num_channel = 3 * (class_number + 5)
    x_scale = MODEL_WIDTH / float(new_w)
    y_scale = MODEL_HEIGHT / float(new_h)
    all_boxes = [[] for ix in range(class_number)]
    for ix in range(3):
        pred = infer_output[2 - ix].reshape((MODEL_HEIGHT // stride_list[ix], \
        MODEL_WIDTH // stride_list[ix], num_channel))
        anchors = anchor_list[ix]
        boxes = decode_bbox(pred, anchors, img_w, img_h, x_scale, y_scale, shift_x_ratio, shift_y_ratio)
        all_boxes = [all_boxes[iy] + boxes[iy] for iy in range(class_number)]

    res = apply_nms(all_boxes, iou_threshold)
    if not res:
        result_return['detection_classes'] = []
        result_return['detection_boxes'] = []
        result_return['detection_scores'] = []
        return result_return
    else:
        new_res = np.array(res)
        picked_boxes = new_res[:, 0:4]
        picked_boxes = picked_boxes[:, [1, 0, 3, 2]]
        picked_classes = convert_labels(new_res[:, 4])
        picked_score = new_res[:, 5]
        result_return['detection_classes'] = picked_classes
        result_return['detection_boxes'] = picked_boxes.tolist()
        result_return['detection_scores'] = picked_score.tolist()
        return result_return

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    #ACL resource initialization
    acl_resource = AclLiteResource()
    acl_resource.init()
    #load model
    model = AclLiteModel(MODEL_PATH)
    images_list_p = [INPUT_DIR_1 + img[:-4] + ".jpg"
                   for img in os.listdir(INPUT_DIR_1)
                   if os.path.splitext(img)[1] in const.IMG_EXT]
    #Read images from the data directory one by one for reasoning
    
    images_list_n = [INPUT_DIR_2 + img[:-4] + ".jpg"
                   for img in os.listdir(INPUT_DIR_2)
                   if os.path.splitext(img)[1] in const.IMG_EXT]
    
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for pic in images_list_p:#这是正例
        #read image
        bgr_img = cv.imread(pic)
        #preprocess
        data, orig = preprocess(pic)
        #Send into model inference
        result_list = model.execute([data,])    
        #Process inference results
        result_return = post_process(result_list, orig)
        #print("result = ", result_return['detection_classes'])
        print("result_boxes =",result_return['detection_boxes'])
        print("result_classes = ",result_return['detection_classes'])
        #那个detection_scores又是什么？是置信度
        if 'mask' in result_return['detection_classes']:
            print("{} 检测结果：正确佩戴口罩".format(pic))
            #开始进行检测,读取和当前文件同名的.csv文件
            with open(pic[:-4]+".csv","r") as f:
                data = csv.reader(f)
                res = list(data)
                real_bbox = [int(i) for i in res[0]]
                IOU = cal_iou(result_return['detection_boxes'][1], real_bbox)    
                if IOU > 0.5:
                    TP = TP+1
                else:
                    FP = FP+1
        else:
            print("{} 检测结果：未正确佩戴口罩".format(pic))
            FP = FP+1
        for i in range(len(result_return['detection_classes'])):
            box = result_return['detection_boxes'][i]
            class_name = result_return['detection_classes'][i]
            confidence = result_return['detection_scores'][i]#这个应该是置信度，但不知为何比较低。不用这个作为判断依据。#可能是因为输入了没戴口罩的数据？
            cv.rectangle(bgr_img, (int(box[1]), int(box[0])), (int(box[3]), int(box[2])), colors[i%6])
            p3 = (max(int(box[1]), 15), max(int(box[0]), 15))
            out_label = class_name            
            cv.putText(bgr_img, out_label, p3, cv.FONT_ITALIC, 0.6, colors[i % 6], 1)
        
        output_file = os.path.join(OUTPUT_DIR, "out_" + os.path.basename(pic))
        print("output path:%s\n" % output_file)
        
        # cv.imwrite(output_file, bgr_img)
    print("start to detect the negative examples")

    for pic in images_list_n:#这里都是负例
        #read image
        bgr_img = cv.imread(pic)
        #preprocess
        data, orig = preprocess(pic)
        #Send into model inference
        result_list = model.execute([data,])    
        #Process inference results
        result_return = post_process(result_list, orig)
        print("result boxs = ", result_return['detection_boxes'])
        print("result classes = ",result_return['detection_classes'])
        #那个detection_scores又是什么？是置信度
        if 'mask' in result_return['detection_classes']:#预测错误
            FN = FN+1
            print("{} 检测结果：正确佩戴口罩\n".format(pic))
        else:
            print("{} 检测结果：错误佩戴口罩\n".format(pic))
            TN = TN+1
        for i in range(len(result_return['detection_classes'])):
            box = result_return['detection_boxes'][i]
            class_name = result_return['detection_classes'][i]
            confidence = result_return['detection_scores'][i]
            cv.rectangle(bgr_img, (int(box[1]), int(box[0])), (int(box[3]), int(box[2])), colors[i%6])
            p3 = (max(int(box[1]), 15), max(int(box[0]), 15))
            out_label = class_name            
            cv.putText(bgr_img, out_label, p3, cv.FONT_ITALIC, 0.6, colors[i % 6], 1)
        
        output_file = os.path.join(OUTPUT_DIR, "out_" + os.path.basename(pic))
        print("output:%s\n" % output_file)
        #cv.imwrite(output_file, bgr_img)
        
    precision = TP / (TP+FP)
    recall = TP/(TP+FN)
    
    print("Precision of the detection task is {:.2}".format(precision))
    print("Recall of the detection task is {:.2}".format(recall))
    print("Accuracy of the detection task is {:.2}".format((TP+TN)/(TP+TN+FP+FN)))
    print("f1-score is {:.2}".format((2*precision*recall)/(precision+recall)))
    print("Execute end")

if __name__ == '__main__':
    main()
 
