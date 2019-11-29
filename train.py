import os
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Model
from keras.layers import Flatten, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
#from keras.layers.convolutional import MaxPooling2D
#from keras.initializers import TruncatedNormal
#from keras.layers.core import Activation
#from keras.layers.core import Flatten
#from keras.layers.core import Dropout


files_train = 0

cwd = os.getcwd()
folder = 'C:/train_data/train'
for sub_folder in os.listdir(folder):
    path, dirs, files = next(os.walk(os.path.join(folder,sub_folder)))
    files_train += len(files)




print(files_train)

img_width, img_height = 150, 150
train_data_dir = "C:/train_data/train"
nb_train_samples = files_train
batch_size = 32
epochs = 20
num_classes = 8

#模型搭建
model = applications.VGG16(weights='imagenet', include_top=False, input_shape = (img_width, img_height, 3))


for layer in model.layers[:10]:
    layer.trainable = False


x = model.output
x = Flatten()(x)
predictions = Dense(num_classes, activation="softmax")(x)


model_final = Model(input = model.input, output = predictions)

#编译模型
model_final.compile(loss = "categorical_crossentropy", 
                    optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), 
                    metrics=["accuracy"]) 

#图片预处理
#数据增强
train_datagen = ImageDataGenerator(
                    rescale = 1./255,
                    horizontal_flip = True,
                    fill_mode = "nearest",
                    zoom_range = 0.1,
                    width_shift_range = 0.1,
                    height_shift_range=0.1,
                    rotation_range=5)


# 创建生成器
train_generator = train_datagen.flow_from_directory(
                    train_data_dir, # 目标目录
                    target_size = (img_height, img_width),# 调整图像尺寸
                    batch_size = batch_size,
                    class_mode = "categorical",
                    classes=['澳大利亚人种', '白色人种', '波利尼西亚人种', '黑色人种',
                     '黄色人种', '密克罗尼西亚人种', '印第安人种', '印度人种'])




checkpoint = ModelCheckpoint("race6.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')



# 使用批量生成器拟合模型
history_object = model_final.fit_generator(
                        train_generator,
                        samples_per_epoch = nb_train_samples,
                        epochs = epochs,

                        callbacks = [checkpoint, early])


model_final.save('race6.h5')
