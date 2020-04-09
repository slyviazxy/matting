import cv2
import time
import numpy as np

def handle_mouse(event,x,y,flags,param):
    global width,height,p##使用global才能在主程序当中改变width,height,从而调整大小
 
       
    if event==cv2.EVENT_MOUSEMOVE and flags==1:##捕捉鼠标移动
     

        ix, iy = x, y  
        cv2.circle(img,(ix,iy),1,(0,0,0),2)##期望刷新点的频率更高一点
        cv2.circle(img,(ix,iy),1,(0,0,0),2)
        cv2.circle(img,(ix,iy),1,(0,0,0),2)

    if event == cv2.EVENT_MOUSEWHEEL:  # 滚轮
       
        if flags > 0:  # up
        
            width=int(width*p)
            height=int(height*p)
      
    
        else:  # down
            height=int(height/p)
            width=int(width/p)          
      
   
        
    
       


origin_photo=(r'C:\Users\XPS\Pictures\Saved Pictures\pcy\image1.jpg')
back_photo=r'C:\Users\XPS\Pictures\Saved Pictures\pcy\image4.jpg'

width=500
height=400
p=5/4## 放大倍数，自行设定
img = cv2.imread(origin_photo)  #加载图片
img= cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)#调整图像大小
back_photo = cv2.imread(back_photo)##作为替换背景的背景图片





copy=img.copy()
cv2.namedWindow('image')

cv2.setMouseCallback('image', handle_mouse)##调用回调方法，一般的鼠标和键盘相应在这里

while(1):##这里实际是捕捉鼠标行为，然后一直刷新，因此调整大小只有这里才生效，因为他覆盖了新的img，而circle则是指向同一个
    img=cv2.resize(img,(width,height))
    
    cv2.imshow('image', img)
    if cv2.waitKey(20) & 0xFF == 27:
        break
    if cv2.waitKey(20) & 0xFF == 13:
        print('输入enter之后')
        copy=img.copy()
        for y in range(width):
            for x in range(height):
                if copy[x,y][0]==0 and copy[x,y][1]==0 and copy[x,y][2]==0:#如果等于黑色pass,其他变成白色
                    pass
                else:
                    copy[x,y]=[255,255,255]
        

        mask = np.zeros([height+2, width+2], np.uint8)#设置掩码层
        # mask = np.zeros([height+2, width+2], np.uint8)

        kernel= cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))##cv2.MORPH_RECT 方法表示
        #getStructuringElement函数会返回指定形状和尺寸的结构元素
        fore_white=cv2.erode(copy, kernel, 4)
        
 
       
        cv2.floodFill(fore_white, mask, (0,0), (0, 0, 0), (50, 50, 50), (30, 30, 30), cv2.FLOODFILL_FIXED_RANGE)##泛洪填色
        

        cv2.imshow('give color to see see',fore_white)
        back_photo = cv2.resize(back_photo, (width, height))
        back_photo = cv2.GaussianBlur(back_photo, (0, 0), 20)##模糊背景图片
       
        fore_white = fore_white/255.0##数据计算
        p =  fore_white[...]
        print(type(copy),len(copy))
        print(type(back_photo)),len(back_photo)

        fusion = p* (img.astype(np.float32)) +(1 - p) * (back_photo.astype(np.float32))

        # fusion = np.dot(p, (copy.astype(np.float32))) +np.dot((1 - p) * (back_photo.astype(np.float32)))#Alpha融合是一个升级版的cut-and-paste，如果表示为公式的话是这个样子：

        #                                                                             #output = foreground * mask + background * (1-mask
        cv2.imshow("fusion", fusion.astype(np.uint8))
    



  
       
        if cv2.waitKey(20) & 0xFF == 27:
            break
       


cv2.destroyAllWindows()




