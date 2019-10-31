import datetime

import cv2  # pip install opencv-python
import numpy as np
import easygui

from wx_notice import send_msg

camera = cv2.VideoCapture(0)  # cv2获取摄像头，0代表默认的摄像头，传给camera这个变量

if not camera.isOpened():
    easygui.msgbox('please turn on you camera or switch to functional one')

'''
取得一个结构化的元素，用来做形态学膨胀， MORPH_ELLIPSE椭圆
'''
es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 4))
# kernel = np.ones((5, 5), np.uint8)  # 数组，用来处理
background = None  # 背景指定为无，在循环中获取第一帧
has_sended_msg = False  # 短信发送，只发送一次

while True:  # 开启一个while循环，一直运行
    has_some_one_in = False  # 标注检测状态
    grabbed, frame_lwpCV = camera.read()  # 从camera中拿取当前帧
    '''
    转换成灰度图像（单通道）
    熟悉图像的同学都知道，一般的图像通常都是3个通道，rgb，三个颜色之间没有必然的联系，
    然后3个通道的图像，进行处理的时候，计算量会比较大，而且代码也会更复杂，这这两方面是不如单通道的。
    '''

    # print(frame_lwpCV)
    gray_lwpCV = cv2.cvtColor(frame_lwpCV, cv2.COLOR_BGR2GRAY)  # 转换成灰度图像(单通道)
    # print(gray_lwpCV)
    '''
    差分法，比较不同，噪声也是不同，所以我们用高斯模糊去掉噪声，
    其实我们都是到高斯模糊就是一种低通滤波器，可以额消除图像里面高频部分，就是噪点的部分

    高斯滤波，其实就是消除噪点，即图像里面高频的部分
    传入灰度图像，
    高斯分布的 高斯和(25, 25)
    高斯分布的 sigma码是3
    高斯图像是有一个分布的，正态分布大家应该有听过说吧，sigma越大，中间部分就约尖
    '''
    gray_lwpCV = cv2.GaussianBlur(gray_lwpCV, (25, 25), 3)  # 高斯滤波，消除噪点
    '''
    使用第一帧作为背景图像，查分法，摄像头开启那一瞬间，博物馆。。
    '''
    # 使用摄像头获取的第一帧作为背景图像
    if background is None:
        background = gray_lwpCV
        continue

    '''
    absolutediff
    一个数字图像是有很多像素
    每个像素都是有数值的
    那么我们这个方法其实就是把每个像素的数值，求一个差值，然后求一个绝对值
    背景帧和当前帧进行比较
    '''
    # 应用差分法
    diff = cv2.absdiff(background, gray_lwpCV)
    '''
    取得threshold
    设定预值
    就是我们说差多少算一样，差多少算不一样是吧，超过50算不一样
    255 白色 显示白色的阈值
    '''
    diff = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)[1]
    '''
    差分已经完成
    形态学膨胀
    给大家解释一下什么是形态学膨胀
    我这里移动，大家看到我们手指，下面白色的部分，实际上比我的手指要大是吧
    有时候我们的差异很小，可以通过形态学膨胀优化这种小的差值，使之整体的连贯起来    
    '''
    diff = cv2.dilate(diff, es, iterations=3)
    '''
    Contours(不连续的物体)
    
    接下来是学opencv都要掌握的一个方法
    findContours 发现图像当中有多少个连续的物体
    RETR_EXTERNAL # 外部轮廓
    CHAIN_APPROX_SIMPLE # 连续的
    '''
    # 使用findContours查找图像中所有连续的物体
    image, contours, hierarchy = cv2.findContours(diff.copy(),
                                                  cv2.RETR_EXTERNAL,
                                                  cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) < 8000:  # 连续的物体太小了就不显示，比如我扔一支笔就不显示了，但我的头太大就显示
            continue
        (x, y, w, h) = cv2.boundingRect(c)  # contours的坐标(左上角)与宽和高
        # 在图像上画框 左上角 右下角， 颜色， 框的字体大小
        cv2.rectangle(frame_lwpCV, (x, y), (x + w, y + h), (0, 255, 0), 1)
        has_some_one_in = True
    # 至此程序主体已经完成

    # 接下来就是通知
    if has_some_one_in and not has_sended_msg:
        send_msg()
        has_sended_msg = True

    cv2.imshow('contours', frame_lwpCV)
    cv2.imshow('dis', diff)

    key = cv2.waitKey(1) & 0xFFf
    if key == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
