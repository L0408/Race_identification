
import numpy as np
from keras.models import load_model
import cv2
from aip import AipFace
import base64
import os
from PIL import Image, ImageFont, ImageDraw



def paint_chinese_opencv(im,chinese,position,fontsize,color):#opencv输出中文
    img_PIL = Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))# 图像从OpenCV格式转换成PIL格式
    font = ImageFont.truetype('simhei.ttf',fontsize,encoding="utf-8")
    #color = (255,0,0) # 字体颜色
    #position = (100,100)# 文字输出位置
    draw = ImageDraw.Draw(img_PIL)
    draw.text(position,chinese,font=font,fill=color)# PIL图片上打印汉字 # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
    img = cv2.cvtColor(np.asarray(img_PIL),cv2.COLOR_RGB2BGR)# PIL图片转cv2 图片
    return img



APP_ID = '17341168'
API_KEY = '1LTA0oHg5il2cziGhoqoGas4'
SECRET_KEY = 'GpHnTOpUrwuRRPqspv204HtolCDzq9a4'

client = AipFace(APP_ID, API_KEY, SECRET_KEY)


races={}
races[0]='澳大利亚人种'
races[1]='白色人种'
races[2]='波利尼西亚人种'
races[3]='黑色人种'
races[4]='黄色人种'
races[5]='密克罗尼西亚人种'
races[6]='印第安人种'
races[7]='印度人种'




for i in range(18):

	with open('C:/test/test1/test'+str(i)+'.jpg', "rb") as f:
	    image = base64.b64encode(f.read())
	    base64_str = str(image,"utf-8")


	imageType = "BASE64"

	""" 调用人脸检测 """

	client.detect(base64_str, imageType)

	""" 如果有可选参数 """
	options = {}
	options["face_field"] = "age,beauty,expression,face_shape,gender,glasses,landmark,landmark72,landmark150,race,quality,eye_status,emotion,face_type"
	options["max_face_num"] = 2
	options["face_type"] = "LIVE"
	options["liveness_control"] = "LOW"
	try:


		a = client.detect(base64_str, imageType, options)['result']

			#print(a)
		location = a['face_list'][0]['location']
		#print(i)



		#print(location)
		
	except TypeError:
		text = '未识别出人像.'

		print(text)

		continue
	

	x = int(location['left'])
	y = int(location['top'])
	w = int(location['width'])
	h = int(location['height'])

	image = cv2.imread('C:/test/test1/test'+str(i)+'.jpg')
	#print(y)

	img = image[int(y*0.38):y+h, x:x+w]


	# cv2.imshow('img', img)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	model = load_model('race6.h5')


	img = cv2.resize(img, (150, 150))
	img = img/255

	img = np.expand_dims(img, axis=0)
	class_predicted = model.predict(img)
	ID = np.argmax(class_predicted)

	text = races[ID]+'的概率: '+str(round(max(class_predicted[0]),2))

	#print(class_predicted)
	print(races[ID], end=': ')
	print(max(class_predicted[0]))

	image = cv2.rectangle(image, (x, int(y*0.4)), ((x+w), y+h), (0, 0, 255), 2)
	image = cv2.resize(image, (512, 512))
	
	image = paint_chinese_opencv(image, text, (0, 0), 30, (255, 0, 0))


	cv2.imshow('image', image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
