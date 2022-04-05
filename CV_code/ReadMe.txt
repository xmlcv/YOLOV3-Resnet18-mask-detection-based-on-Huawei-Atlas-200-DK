本文件夹下保存本项目所用的所有代码。
detect_mask.py为项目主文件，基于华为官方gitee开源代码库的口罩检测样例：
https://gitee.com/ascend/samples/tree/master/python/level2_simple_inference/2_object_detection/YOLOV3_mask_detection_picture的主文件设计，并对模型评价指标和推理结果的输出形式进行了改进。
按照上述链接中的步骤下载源代码，将detect_mask.py放在src目录下，将数据集放在YOLOV3_mask_detection_picture文件夹下，通过python3 detect_mask.py即可运行。

picture_capture.py为数据采集文件，调用电脑摄像头每隔5s进行一次拍摄。

tagging.py为数据标注文件，基于OpenCv库实现手动数据标注。运行代码，将弹出显示指定文件夹下数据的窗口。点击图片上的点，即可显示该点的坐标，通过确定检测框的左上角和右下角像素点坐标进行标注。
标注完一张图片后，点击q键即可退出并切换为下一张图片，同时将上一张图片中标注点的坐标保存为与图片同名的.csv文件。
