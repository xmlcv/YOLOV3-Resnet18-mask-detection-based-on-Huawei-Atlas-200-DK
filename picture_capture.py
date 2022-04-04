import cv2
import time

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW) #设置摄像头 0是默认的摄像头 如果你有多个摄像头的话呢，可以设置1,2,3....
judge = 0

while True:   #进入无限循环
    ret,frame = cap.read() #将摄像头拍到的图像作为frame值
    cv2.imshow('frame',frame) #将frame的值显示出来 有两个参数 前一个是窗口名字，后面是值
    '''
    c = cv2.waitKey(1) #判断退出的条件 当按下'Q'键的时候呢，就退出
	if c == ord('q'):
		break
    '''
    cv2.imwrite("D:/TMP_NO_MASK/wrongmask" + str(judge) + ".jpg", frame)#错误佩戴
    #cv2.imwrite("D:/TMP_HV_MASK/" + "withmask" + str(judge) + ".jpg", frame)#正确佩戴
    print(cap.get(3))
    print(cap.get(4))
    print("save" + str(judge) + ".jpg successfuly!")
    print("-------------------------")
    #后面最好是能拿这个文件夹进行运行
    if judge > 15:
        break
    time.sleep(3)
    judge = judge + 1
    
cap.release()  #常规操作
cv2.destroyAllWindows()