import colorsys
import copy
import os
from timeit import default_timer as timer

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from tensorflow.compat.v1.keras import backend as K
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model, load_model

from nets.yolo4 import yolo_body, yolo_eval
from utils.utils import letterbox_image

from sort import SORT

# change this path to yours, or you can change the "current_path" directly
ros_workspace = "/catkin_ws/src/cv-meter/scripts"
try:
    user_name = os.environ['USERNAME']
    current_path = "/home/"+user_name + ros_workspace
except:
    current_path = os.environ['HOME'] + ros_workspace

image_size = '720p'

fx = 1073.86040929768
fy = 1074.22963341670
cx = 942.114011833120
cy = 554.565831642811
def pixel2world(pix_u, pix_v, depth):
    world_d = depth * 0.001

    world_z = round(float(world_d), 5)
    if image_size == '1080p':
        world_x = round(float((pix_v - cx) * world_z) / fx, 5)
        world_y = round(float((pix_u - cy) * world_z) / fy, 5)
    else:
        world_x = round(float((pix_v * 2 - cx) * world_z) / fx, 5)
        world_y = round(float((pix_u * 2- cy) * world_z) / fy, 5)

    return [world_x, world_y, world_z]

select_area = 'center'
select_d = 'median'
select_uv = 'median'
def get_position(box, depth_pic, image_size):
    top, left, bottom, right = box
    h = bottom - top
    w = right - left
    top = top - 5
    left = left - 5
    bottom = bottom + 5
    right = right + 5
    top = max(0, np.floor(top + 0.5).astype('int32'))
    left = max(0, np.floor(left + 0.5).astype('int32'))
    bottom = min(image_size[1], np.floor(bottom + 0.5).astype('int32'))
    right = min(image_size[0], np.floor(right + 0.5).astype('int32'))

    if select_area == 'center':
        depthbox = depth_pic[top + int(h/4):bottom - int(h/4), left + int(w/5):right - int(w/5)]
    else:
        depthbox = depth_pic[top:bottom, left:right]
    try:
        if select_d == 'min':
            # 使用bbox中深度的最小值代表对象的深度
            depth = np.min(depthbox[np.nonzero(depthbox)])
        elif select_d == 'median':
            # 使用bbox中深度的median代表对象的深度
            depth = np.median(depthbox[np.nonzero(depthbox)])
        else:
            print("Error: unknow parameter for select_d:", select_d)
            # 调用depthbox是获取中心点相对坐标
            depth = depthbox[int((bottom+top)/2),int((right+left)/2)]

        if select_uv == 'center':
            location = pixel2world(int((bottom+top)/2), int((right+left)/2), depth)
        else:
            axis = np.where(depthbox == depth)
            location = pixel2world(int(top+axis[0][0]), left+axis[1][0], depth)
        #if depth == 0:
        #    depth = np.min(depthbox[np.nonzero(depthbox)])
        #    axis = np.where(depthbox == depth)
        #    location = pixel2world(int(top+axis[0][0]), left+axis[1][0], depth)
        #else:
            ## center
            #location = pixel2world(int((bottom+top)/2), int((right+left)/2), depth)
    except:
        location = [0,0,-1]
    return location, top, left, bottom, right


#--------------------------------------------#
#   使用自己训练好的模型预测需要修改2个参数
#   model_path和classes_path都需要修改！
#   如果出现shape不匹配，一定要注意
#   训练时的model_path和classes_path参数的修改
#--------------------------------------------#
class YOLO(object):
    _defaults = {
        #"model_path"        : 'model_data/yolo4_weight.h5',
        "model_path"        : current_path+'/logs/last2.h5',
        "anchors_path"      : current_path+'/model_data/yolo_anchors.txt',
        "classes_path"      :current_path+ '/model_data/my_classes.txt',
        "score"             : 0.5,
        "iou"               : 0.3,
        "eager"             : True,
        "max_boxes"         : 100,
        # 显存比较小可以使用416x416
        # 显存比较大可以使用608x608
        "model_image_size"  : (416, 416),
        #---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        #---------------------------------------------------------------------#
        "letterbox_image"   : False,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化yolo
    #---------------------------------------------------#
    def __init__(self, use_sort=False, **kwargs):
        self.__dict__.update(self._defaults)
        if not self.eager:
            tf.compat.v1.disable_eager_execution()
            self.sess = K.get_session()
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.generate()
        self.use_sort = False
        if use_sort:
            self.use_sort = True
            self.person_sort = SORT(max_disappear=120, max_thr=1.2)
            self.robot_sort = SORT(max_disappear=120, max_thr=1.2)


    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    #---------------------------------------------------#
    #   获得所有的先验框
    #---------------------------------------------------#
    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    #---------------------------------------------------#
    #   载入模型
    #---------------------------------------------------#
    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        #---------------------------------------------------#
        #   计算先验框的数量和种类的数量
        #---------------------------------------------------#
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)

        #---------------------------------------------------------#
        #   载入模型
        #---------------------------------------------------------#
        self.yolo_model = yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
        self.yolo_model.load_weights(self.model_path)

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

        # 打乱颜色
        np.random.seed(10101)
        np.random.shuffle(self.colors)
        np.random.seed(None)

        #---------------------------------------------------------#
        #   在yolo_eval函数中，我们会对预测结果进行后处理
        #   后处理的内容包括，解码、非极大抑制、门限筛选等
        #---------------------------------------------------------#
        if self.eager:
            self.input_image_shape = Input([2,],batch_size=1)
            inputs = [*self.yolo_model.output, self.input_image_shape]
            outputs = Lambda(yolo_eval, output_shape=(1,), name='yolo_eval',
                arguments={'anchors': self.anchors, 'num_classes': len(self.class_names), 'image_shape': self.model_image_size,
                'score_threshold': self.score, 'eager': True, 'max_boxes': self.max_boxes, 'letterbox_image': self.letterbox_image})(inputs)
            self.yolo_model = Model([self.yolo_model.input, self.input_image_shape], outputs)
        else:
            self.input_image_shape = K.placeholder(shape=(2, ))

            self.boxes, self.scores, self.classes = yolo_eval(self.yolo_model.output, self.anchors,
                    num_classes, self.input_image_shape, max_boxes=self.max_boxes,
                    score_threshold=self.score, iou_threshold=self.iou, letterbox_image=self.letterbox_image)

    @tf.function
    def get_pred(self, image_data, input_image_shape):
        out_boxes, out_scores, out_classes = self.yolo_model([image_data, input_image_shape], training=False)
        return out_boxes, out_scores, out_classes

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image, depth_pic=None):
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        if self.letterbox_image:
            boxed_image = letterbox_image(image, (self.model_image_size[1],self.model_image_size[0]))
        else:
            boxed_image = image.convert('RGB')
            boxed_image = boxed_image.resize((self.model_image_size[1],self.model_image_size[0]), Image.BICUBIC)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        #---------------------------------------------------------#
        #   将图像输入网络当中进行预测！
        #---------------------------------------------------------#
        if self.eager:
            # 预测结果
            input_image_shape = np.expand_dims(np.array([image.size[1], image.size[0]], dtype='float32'), 0)
            out_boxes, out_scores, out_classes = self.get_pred(image_data, input_image_shape)
        else:
            # 预测结果
            out_boxes, out_scores, out_classes = self.sess.run(
                [self.boxes, self.scores, self.classes],
                feed_dict={
                    self.yolo_model.input: image_data,
                    self.input_image_shape: [image.size[1], image.size[0]],
                    K.learning_phase(): 0
                })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        #---------------------------------------------------------#
        #   设置字体
        #---------------------------------------------------------#
        font = ImageFont.truetype(font=current_path+'/font/simhei.ttf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = max((image.size[0] + image.size[1]) // 300, 1)

        pop_list = []
        locations = []
        if self.use_sort:
            boxes_person = []
            scores_person = []
            boxes_robot = []
            scores_robot = []
            for i in range(len(out_boxes)):
                if self.class_names[out_classes[i]] == "person":
                    boxes_person.append(out_boxes[i])
                    scores_person.append(out_scores[i])
                    pop_list.append(i)
                elif self.class_names[out_classes[i]] == "manipulator":
                    boxes_robot.append(out_boxes[i])
                    scores_robot.append(out_scores[i])
                    pop_list.append(i)

            self.person_sort.update(boxes_person, scores_person)
            self.robot_sort.update(boxes_robot, scores_robot)

            bboxs_sort, scores_sort = self.person_sort.get_output()
            for i, bbox_pre in enumerate(bboxs_sort):
                location, top, left, bottom, right = get_position(bbox_pre, depth_pic, image.size)
                location.append("person")
                locations.append(location)
                # Draw bbox
                draw = ImageDraw.Draw(image)

                # Draw bbox
                if scores_sort[i] > 0:
                    label = '{} {:.2f} {}{:.2f},{:.2f},{:.2f}{}'.format("Person", scores_sort[i],
                    "Pos:(", location[0],location[1],location[2],")")
                else:
                    label = '{}{:.4f},{:.4f},{:.4f}{}'.format("Person_predict  Pos:(",
                     location[0],location[1],location[2] ,")")
                draw = ImageDraw.Draw(image)
                label_size = draw.textsize(label, font)
                label = label.encode('utf-8')
                print(label, top, left, bottom, right, location)

                if top - label_size[1] >= 0:
                    text_origin = np.array([bbox_pre[1], bbox_pre[0] - label_size[1]])
                else:
                    text_origin = np.array([bbox_pre[1], bbox_pre[0] + 1])

                for k in range(thickness):
                    # Blue rectangle for priori estimate
                    draw.rectangle(
                        [bbox_pre[1]+ k, bbox_pre[0]+ k, bbox_pre[3]+ k, bbox_pre[2]+ k],
                        outline=(255, 0, 255))
                draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + label_size)], fill=(255, 0, 255))
                draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
                del draw

            bboxs_sort, scores_sort = self.robot_sort.get_output()
            for i, bbox_pre in enumerate(bboxs_sort):
                location, top, left, bottom, right = get_position(bbox_pre, depth_pic, image.size)
                location.append("manipulator")
                locations.append(location)

                # Draw bbox
                if scores_sort[i] > 0:
                    label = '{} {:.2f} {}{:.2f},{:.2f},{:.2f}{}'.format("Robot", scores_sort[i],
                    "Pos:(", location[0],location[1],location[2],")")
                else:
                    label = '{}{:.4f},{:.4f},{:.4f}{}'.format("Robot_predict  Pos:(",
                     location[0],location[1],location[2] ,")")
                draw = ImageDraw.Draw(image)
                label_size = draw.textsize(label, font)
                label = label.encode('utf-8')
                print(label, top, left, bottom, right, location)

                if top - label_size[1] >= 0:
                    text_origin = np.array([bbox_pre[1], bbox_pre[0] - label_size[1]])
                else:
                    text_origin = np.array([bbox_pre[1], bbox_pre[0] + 1])

                for k in range(thickness):
                    # Blue rectangle for priori estimate
                    draw.rectangle(
                        [bbox_pre[1]+ k, bbox_pre[0]+ k, bbox_pre[3]+ k, bbox_pre[2]+ k],
                        outline=(255, 0, 0))
                draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + label_size)], fill=(255, 0, 0))
                draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
                del draw

        for i, c in list(enumerate(out_classes)):
            if i in pop_list:
                continue
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            location, top, left, bottom, right = get_position(box, depth_pic, image.size)
            location.append(predicted_class)
            locations.append(location)
            # Draw bbox
            if predicted_class == "manipulator":
                label = '{} {:.2f} {}{:.2f},{:.2f},{:.2f}{}'.format("Robot", score,
                "Pos:(", location[0],location[1],location[2],")")
            elif predicted_class == "person":
                label = '{} {:.2f} {}{:.2f},{:.2f},{:.2f}{}'.format("Person", score,
                "Pos:(", location[0],location[1],location[2] ,")")
            else:
                label = '{} {:.2f} {}{:.2f},{:.2f},{:.2f}{}'.format(predicted_class, score,
                "Pos:(", location[0],location[1],location[2],")")

            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right, location)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for k in range(thickness):
                draw.rectangle(
                    [left + k, top + k, right - k, bottom - k],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        return image, locations

    def close_session(self):
        self.sess.close()
