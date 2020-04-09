
import cv2 as cv2
import numpy as np

origin_photo=r'C:\Users\XPS\Pictures\Saved Pictures\pcy\image1.jpg'##设定前景和背景图片
back_photo=r'C:\Users\XPS\Pictures\Saved Pictures\pcy\image4.jpg'





width=500##定义图像大小，避免过大，并且融合图像的时候前后两张图像保持大小一致
height=400
origin = cv2.imread(origin_photo)##前景图片
origin= cv2.resize(origin, (width, height), interpolation=cv2.INTER_CUBIC)##重新调整大小
back_photo = cv2.imread(back_photo)##作为替换背景的背景图片

x,y,w,h= cv2.selectROI(origin, False)  ##opencv使用坐标都是返回左上角,(x,y)以及宽高(w,h)   selectROI让你选择ROI区域

roi = origin[int(y):int(y+h), int(x):int(x+w)]##
copy = origin.copy()
cv2.rectangle(copy, (int(x), int(y)),(int(x+w), int(y+h)), (255, 0, 0), 3)
bgd_model = np.zeros((1,65),np.float64) # bgd模型数组,np.zeros() 是numpy产生数组的方法
fgd_model = np.zeros((1,65),np.float64) 
mask = np.zeros(origin.shape[:2], dtype=np.uint8)##掩码,自定义的图像矩阵，用来计算图像的像素值
rect = (int(x), int(y), int(w), int(h)) # 包括前景的矩形，格式为(x,y,w,h)
iter_count=11##这个是迭代的次数，越多越准确，但也越慢

cv2.grabCut(origin,mask,rect,bgd_model,fgd_model, iter_count, mode=cv2.GC_INIT_WITH_RECT)

#----------------------------------------
back_photo = cv2.resize(back_photo, (width, height))

fore_white = np.where((mask==1) + (mask==3), 255, 0).astype('uint8')##mask==1表示确定为前景的标示区域，mask==3表示可能前景标示，
                                                                ##np.where()满足条件输出255，不满足输出0,.astype('uint8')数值转换问题，

cv2.imshow('no_blur_mask',fore_white)#‘blur_mask’和'force_blur'比较就知道为什么使用膨胀和模糊了

kernel= cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))##cv2.MORPH_RECT 方法表示
cv2.dilate(fore_white, kernel, fore_white)##fore_white表示这是一个前景提取后生成的白色掩码，因为与图像融合的时候，会两个相同位置像素颜色加权计算，
                                           ##白色掩膜就可以遮盖背景图片的相同部分，前景就可以正确的映射到新图片上
fore_white = cv2.medianBlur(fore_white, 9)
cv2.imshow('blur_mask',fore_white)

back_photo = cv2.GaussianBlur(back_photo, (0, 0), 20)##模糊背景图片
fore_white = fore_white/255.0##数据计算
p =  fore_white[..., None]##NONE补充一个通道，print就清楚了

fusion = p* (origin.astype(np.float32)) +(1 - p) * (back_photo.astype(np.float32))#Alpha融合是一个升级版的cut-and-paste，如果表示为公式的话是这个样子：

                                                                             #output = foreground * mask + background * (1-mask
cv2.imshow("fusion", fusion.astype(np.uint8))


cv2.waitKey(0)
cv2.destroyAllWindows()