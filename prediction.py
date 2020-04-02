# -*- coding: utf-8 -*
import os
from path import MODEL_PATH, DATA_PATH
import numpy as np
from flyai.framework import FlyAI
from keras.models import load_model
import cv2
import pandas as pd

model_path = os.path.join(MODEL_PATH, 'my_model.h5')

class Prediction(FlyAI):
    def __init__(self):
        self.model = None
        print("init Prediction")

    def load_model(self):
        '''
        模型初始化，必须在构造方法中加载模型
        '''
        print("load model")
        if self.model == None:
            self.model = load_model(model_path)


    def predict(self, input_data):
        '''
        模型预测返回结果
        :param input: input是app.yaml中 model-》input设置的输入，项目设定不可以修改，输入一条数据是字典类型
        :return: 返回预测结果，是app.yaml中 model-》output设置的输出
        '''
        # {{'image_path': 'image/imgs5effcd4a0e981fbc63cc0f24.png'}
        print(os.path.join(DATA_PATH , 'FaceLandmarkLocalization',input_data['image_path']))
        img = cv2.imread(os.path.join(DATA_PATH, 'FaceLandmarkLocalization', input_data['image_path']))
        height = img.shape[0]
        width = img.shape[1]
        half_height = height / 2.0
        half_width = width / 2.0
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        batch_x = np.reshape(img, (-1, img.shape[0], img.shape[1], img.shape[2]))

        pred = self.model.predict(batch_x)[0]
        # 把pred预测的结果返回到原始大小
        left_eye_pupil_radius = pred[-2] * half_width + half_width
        right_eye_pupil_radius = pred[-1] * half_width + half_width
        label_xy = np.reshape(np.array(pred[:-2]), (-1, 2))
        label_xy[:, 0] = label_xy[:, 0] * half_width + half_width
        label_xy[:, 1] = label_xy[:, 1] * half_height + half_height

        face_hairline_xy = label_xy[:145]  # 145
        face_contour_left_xy = label_xy[145:145 + 64]  # 64
        face_contour_right_xy = label_xy[209:209 + 64]  # 64
        left_eyebrow_xy = label_xy[273:273 + 64]  # 64
        right_eyebrow_xy = label_xy[337:337 + 64]  # 64
        left_eye_xy = label_xy[401:401 + 63]  # 63
        left_eye_pupil_center_xy = label_xy[464:464 + 1]  # 1
        left_eye_eyelid_xy = label_xy[465:465 + 64]  # 64
        right_eye_xy = label_xy[529:529 + 63]  # 63
        right_eye_pupil_center_xy = label_xy[592:592 + 1]  # 1
        right_eye_eyelid_xy = label_xy[593:593 + 64]  # 64
        nose_left_xy = label_xy[657:657 + 63]  # 63
        nose_right_xy = label_xy[720:720 + 63]  # 63
        nose_midline_xy = label_xy[783:783 + 60]  # 60
        left_nostril_xy = label_xy[843:843 + 1]  # 1
        right_nostril_xy = label_xy[844:844 + 1]  # 1
        upper_lip_xy = label_xy[845:845 + 64]  # 64
        lower_lip_xy = label_xy[909: 909 + 32]  # 32



        return [left_eye_pupil_radius, right_eye_pupil_radius, left_eye_pupil_center_xy,right_eye_pupil_center_xy,
                face_hairline_xy, face_contour_left_xy, face_contour_right_xy,
                left_eyebrow_xy, right_eyebrow_xy,
                left_eye_xy , left_eye_eyelid_xy,
                right_eye_xy, right_eye_eyelid_xy,
                nose_left_xy, nose_right_xy, nose_midline_xy,
                left_nostril_xy, right_nostril_xy,
                upper_lip_xy, lower_lip_xy]

if __name__ == '__main__':
    # 读取数据
    df = pd.read_csv(os.path.join(DATA_PATH, 'FaceLandmarkLocalization/train.csv'))

    df['label'] = df['label'].apply(eval)  # 转成dict类型
    image_path_list = df['image_path'].values
    label_list = df['label'].values

    # image/11500.png
    # b= {'image_path': './data/input/FaceLandmarkLocalization_FlyAI/image/imgsbf2908dfa55a56c8ce1b2dab.png'}
    b = {'image_path': 'image/imgsbf2908dfa55a56c8ce1b2dab.png'}
    # b = "FishClassification/image/0.png"
    a = Prediction()
    a.load_model()
    c = a.predict(b)
    print(c)