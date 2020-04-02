# -*- coding: utf-8 -*-
from flyai.data_helper import DataHelper
from flyai.framework import FlyAI
import pandas as pd
from path import DATA_PATH, MODEL_PATH
import os
import argparse
import cv2
import numpy as np
from keras.applications import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from flyai.utils import remote_helper

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
model_path = os.path.join(MODEL_PATH, 'my_model.h5')

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=5, type=int, help="batch size")
args = parser.parse_args()


# 下载数据
data_helper = DataHelper()
data_helper.download_from_ids("FaceLandmarkLocalization") # ./data/input/PostRecommendation

# 读取数据
df = pd.read_csv(os.path.join(DATA_PATH, 'FaceLandmarkLocalization/train.csv'))

df['label'] = df['label'].apply(eval) # 转成dict类型
image_path_list = df['image_path'].values
label_list = df['label'].values

# ##################################  下面提供一种数据处理的方法 ###################################

class Main(FlyAI):
    def get_line(self, array, label, name):
        for i in range(len(array)):
            str_name = name + '_' + str(i)
            array[i][0] = label[str_name]['x']
            array[i][1] = label[str_name]['y']

    def get_batch_data(self, image_path_list, label_list, batch_size):
        batch_x = []
        batch_y = []
        while len(batch_x) < batch_size:
            index = np.random.randint(len(image_path_list))
            img = cv2.imread(os.path.join(DATA_PATH, 'FaceLandmarkLocalization', image_path_list[index]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height = img.shape[0]
            width = img.shape[1]

            mark = label_list[index]['landmark']
            face = mark['face']
            left_eyebrow = mark['left_eyebrow']
            right_eyebrow = mark['right_eyebrow']
            right_eye_eyelid = mark['right_eye_eyelid']
            left_eye_eyelid = mark['left_eye_eyelid']
            right_eye = mark['right_eye']
            left_eye = mark['left_eye']
            mouth = mark['mouth']
            nose = mark['nose']

            face_hairline_xy = np.zeros((145, 2))
            self.get_line(face_hairline_xy, face, 'face_hairline')
            face_contour_left_xy = np.zeros((64, 2))
            self.get_line(face_contour_left_xy, face, 'face_contour_left')
            face_contour_right_xy = np.zeros((64, 2))
            self.get_line(face_contour_right_xy, face, 'face_contour_right')

            left_eyebrow_xy = np.zeros((64, 2))
            self.get_line(left_eyebrow_xy, left_eyebrow, 'left_eyebrow')

            right_eyebrow_xy = np.zeros((64, 2))
            self.get_line(right_eyebrow_xy, right_eyebrow, 'right_eyebrow')

            left_eye_xy = np.zeros((63, 2))
            self.get_line(left_eye_xy, left_eye, 'left_eye')

            left_eye_pupil_center_xy = np.zeros((1, 2))
            left_eye_pupil_center_xy[0, 0] = left_eye['left_eye_pupil_center']['x']
            left_eye_pupil_center_xy[0, 1] = left_eye['left_eye_pupil_center']['y']

            left_eye_pupil_radius = left_eye['left_eye_pupil_radius']

            left_eye_eyelid_xy = np.zeros((64, 2))
            self.get_line(left_eye_eyelid_xy, left_eye_eyelid, 'left_eye_eyelid')

            right_eye_xy = np.zeros((63, 2))
            self.get_line(right_eye_xy, right_eye, 'right_eye')

            right_eye_pupil_center_xy = np.zeros((1, 2))
            right_eye_pupil_center_xy[0, 0] = right_eye['right_eye_pupil_center']['x']
            right_eye_pupil_center_xy[0, 1] = right_eye['right_eye_pupil_center']['y']

            right_eye_pupil_radius = right_eye['right_eye_pupil_radius']

            right_eye_eyelid_xy = np.zeros((64, 2))
            self.get_line(right_eye_eyelid_xy, right_eye_eyelid, 'right_eye_eyelid')

            nose_left_xy = np.zeros((63, 2))
            self.get_line(nose_left_xy, nose, 'nose_left')

            nose_right_xy = np.zeros((63, 2))
            self.get_line(nose_right_xy, nose, 'nose_right')

            nose_midline_xy = np.zeros((60, 2))
            self.get_line(nose_midline_xy, nose, 'nose_midline')

            left_nostril_xy = np.zeros((1, 2))
            left_nostril_xy[0, 0] = nose['left_nostril']['x']
            left_nostril_xy[0, 1] = nose['left_nostril']['y']

            right_nostril_xy = np.zeros((1, 2))
            right_nostril_xy[0, 0] = nose['right_nostril']['x']
            right_nostril_xy[0, 1] = nose['right_nostril']['y']

            upper_lip_xy = np.zeros((64, 2))
            self.get_line(upper_lip_xy, mouth, 'upper_lip')

            lower_lip_xy = np.zeros((32, 2))
            self.get_line(lower_lip_xy, mouth, 'lower_lip')

            all_xy = np.concatenate((face_hairline_xy, face_contour_left_xy, face_contour_right_xy,
                                     left_eyebrow_xy, right_eyebrow_xy,
                                     left_eye_xy, left_eye_pupil_center_xy, left_eye_eyelid_xy,
                                     right_eye_xy, right_eye_pupil_center_xy, right_eye_eyelid_xy,
                                     nose_left_xy, nose_right_xy, nose_midline_xy,
                                     left_nostril_xy, right_nostril_xy,
                                     upper_lip_xy, lower_lip_xy))

            # 归一化坐标
            half_height = height / 2.0
            half_width = width / 2.0

            all_xy[:, 0] = (all_xy[:, 0] - half_width) / half_width  # 这里x表示距离图片左上角的left值
            all_xy[:, 1] = (all_xy[:, 1] - half_height) / half_height  # 这里y表示距离图片左上角的top值
            left_eye_pupil_radius = (left_eye_pupil_radius - half_width) / half_width
            right_eye_pupil_radius = (right_eye_pupil_radius - half_width) / half_width

            # 整合成
            label = np.concatenate(
                (np.reshape(all_xy, (-1)), [left_eye_pupil_radius, right_eye_pupil_radius]))  # 1884
            img = cv2.resize(img, (256, 256))
            batch_x.append(img)
            batch_y.append(label)

        return np.array(batch_x), np.array(batch_y)

    def train(self):
        # 构建模型
        batch_steps = len(label_list) // args.BATCH
        # 构建不带分类器的预训练模型
        base_model = InceptionV3(weights=None, include_top=False)
        path = remote_helper.get_remote_date(
            'https://www.flyai.com/m/v0.5|inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')
        base_model.load_weights(path)

        # 添加全局平均池化层
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # 添加一个全连接层
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(1884)(x)
        # 构建我们需要训练的完整模型
        model = Model(inputs=base_model.input, outputs=predictions)
        # 编译模型
        model.compile(optimizer='rmsprop', loss='mse')
        print('model done!!!')

        min_loss = 100
        for epoch in range(args.EPOCHS):
            loss_50 = []
            for i in range(batch_steps):
                now_step = epoch * batch_steps + i
                batch_x, batch_y = self.get_batch_data(image_path_list, label_list, args.BATCH)
                loss = model.train_on_batch(batch_x, batch_y)
                print('epoch: %d/%d, batch: %d/%d, loss: %f/%f' % (epoch, args.EPOCHS, i, batch_steps, loss, min_loss))
                loss_50.append(loss)
                if now_step % 50 == 0:
                    mean_loss = np.mean(np.array(loss_50))
                    loss_50 = []
                    if mean_loss < min_loss:
                        min_loss = mean_loss
                        model.save(model_path)
                        print('saved model!!!')


main = Main()
main.train()