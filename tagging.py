import cv2
import numpy as np
import csv

import os
filenames = os.listdir(r"D:\TMP_HV_MASK")
print(filenames)

for i in filenames:
    file = "D:\\TMP_HV_MASK"+'\\' + i
    save_file = "D:\\TMP_HV_MASK" + "\\" + i[:-4] + '.csv'
    print(file)
    print(save_file)
    
    img = cv2.imread("D:\\TMP_HV_MASK"+'\\'+i)
    #img=cv2.imread('out_9.jpg')
    L = []
    def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            xy = "%d,%d" % (x, y)
            cv2.circle(img, (x, y), 1, (255, 0, 0), thickness = -1)
            cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0,0,0), thickness = 1)
            print((x,y))#然后按y,x的顺序保存
            L.append(y)
            L.append(x)
            cv2.imshow("image", img)
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
    #现在这个算法已经能得到指定点的坐标，同时输出结果，后续也可以方便地保存。还有一个问题，每个坐标点都有x和y两个坐标，但算法本身给出的似乎是只有一个值？这个应该如何确定？
    while(1):
        cv2.imshow("image", img)
       
        c = cv2.waitKey(0) #判断退出的条件 当按下'Q'键的时候呢，就退出
        if c == ord('q'):
            break
    cv2.destroyAllWindows()

    out = open("D:\\TMP_HV_MASK"+'\\' + i[:-4]+".csv","a")
    csv_writer = csv.writer(out)
    csv_writer.writerow(L)
    