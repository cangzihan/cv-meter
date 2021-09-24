from KalmanFilter import KalmanFilter
import numpy as np
import cv2


class KalmanTracker(KalmanFilter):
    def __init__(self, initial_position=(0, 0, 0, 0)):
        super().__init__()
        self.age = 0
        self.disappear = 0

        self.P = np.identity(7) * 10
        self.P[4, 4] = 1000
        self.P[5, 5] = 1000
        self.P[6, 6] = 1000
        self.Q = np.identity(7)
        self.Q[4, 4] = 0.025
        self.Q[5, 5] = 0.025
        self.Q[6, 6] = 0.00001
        self.R = np.identity(4)
        self.R[2, 2] = 10
        self.R[3, 3] = 10
        self.H = np.zeros((4, 7))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1
        self.H[3, 3] = 1
        self.A = np.identity(7)
        self.A[0, 4] = self.delta_t
        self.A[1, 5] = self.delta_t
        self.A[2, 6] = self.delta_t * 0.0001
        self.A_predict = np.identity(7)
        self.A_predict[0, 4] = self.delta_t / 2
        self.A_predict[1, 5] = self.delta_t / 2
        self.A_predict[2, 6] = self.delta_t / 2 * 0.0001

        self.x_before = np.array([[initial_position[0], initial_position[1], initial_position[2], initial_position[3],
                                   0, 0, 0]])
        self.x_before = self.x_before.transpose()
        self.save_list = range(4)  # Save 4 parameters [u, v, s, r]


def cal_ciou(bbox1, bbox2):
    top1, left1, button1, right1 = bbox1
    top2, left2, button2, right2 = bbox2
    C_w = max(right1, right2) - min(left1, left2)
    C_h = max(button1, button2) - min(top1, top2)
    w1 = right1-left1
    w2 = right2-left2
    h1 = button1-top1
    h2 = button2-top2
    center1_x = (right1 + left1) / 2
    center1_y = (button1 + top1) / 2
    center2_x = (right2 + left2) / 2
    center2_y = (button2 + top2) / 2
    d_2 = (center1_x-center2_x) ** 2 + (center1_y-center2_y) ** 2
    c_2 = C_w ** 2 + C_h ** 2
    area_1 = w1 * h1
    area_2 = w2 * h2
    sum_area = area_1 + area_2
    w_inter = max(0, min(left1, left2) + w1 + w2 - max(right1, right2))
    h_inter = max(0, min(top1, top2) + h1 + h2 - max(button1, button2))
    area_inter = w_inter * h_inter

    iou = area_inter / (sum_area - area_inter)
    diou = iou - d_2 / c_2
    v = 4.0 * ((np.arctan(w1/h1) - np.arctan(w2/h2)) ** 2) / (np.pi ** 2)
    if v != 0:
        a = v / (1 - iou + v)
    else:
        a = 0
    ciou = diou - a * v
    return ciou


def get_ciou_loss(bbox1, bbox2):
    return 1 - cal_ciou(bbox1, bbox2)


# [top, left, button, right, score] --> [u, v, s, r]
def bbox_2_z(bbox):
    x = (bbox[1] + bbox[3]) / 2
    y = (bbox[0] + bbox[2]) / 2
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    s = w * h
    r = w / h
    return [x, y, s, r]


def z_2_bbox(z):
    w = np.sqrt(z[2] * z[3])
    h = np.sqrt(z[2] / z[3])
    top = int(z[1] - w/2)
    buttom = int(z[1] + w/2)
    left = int(z[0] - h/2)
    right = int(z[0] + h/2)
    return [top, left, buttom, right]


# Hungarian method
# If you have installed the "sklearn" package. The following function can be replaced by "linear_assignment".
def hungarian(input_array):
    n = np.max(input_array.shape)

    if input_array.shape[0] > input_array.shape[1]:
        add_col = np.zeros((input_array.shape[0], input_array.shape[0]-input_array.shape[1]))
        input_array_added = np.column_stack((input_array, add_col))

        # Step 1
        min_row = np.min(input_array_added, axis=1)
        min_row.resize((len(min_row), 1))
        hungarian_array = input_array_added - min_row
    else:
        # Step 1
        min_row = np.min(input_array, axis=1)
        min_row.resize((len(min_row), 1))
        hungarian_array = input_array - min_row

    if input_array.shape[0] < input_array.shape[1]:
        add_row = np.zeros((input_array.shape[1]-input_array.shape[0], input_array.shape[1]))
        hungarian_array = np.row_stack((hungarian_array, add_row))

    # Step 2
    min_col = np.min(hungarian_array, axis=0)
    min_col.resize((1, len(min_col)))
    hungarian_array -= min_col

    while True:
        # Step 3
        zero_index = np.where(hungarian_array == 0)

        zero_count_row = []
        zero_count_col = []
        for i in range(len(hungarian_array)):
            zero_count_row.append(np.sum(hungarian_array[i] == 0))
        for i in range(len(hungarian_array[0])):
            zero_count_col.append((np.sum(hungarian_array[:, i] == 0)))

        indepent_elements = []
        zero_count_dict = {}
        for i in range(len(zero_index[0])):
            temp = (zero_index[0][i], zero_index[1][i])
            zero_count_dict[temp] = zero_count_row[zero_index[0][i]]+zero_count_col[zero_index[1][i]]-1

        zero_coordinate = list(zero_count_dict.keys())
        while len(zero_count_dict) > 0:
            indepent_elements.append(min(zero_count_dict.keys(), key=(lambda x: zero_count_dict[x])))
            r = indepent_elements[-1][0]
            c = indepent_elements[-1][1]
            keys = list(zero_count_dict.keys())
            for k in keys:
                if r == k[0] or c == k[1]:
                    zero_count_dict.pop(k)

        row_correct = [i for i in range(len(hungarian_array)) if i not in [j[0] for j in indepent_elements]]
        col_correct = []

        unsolved_row_correct = [i for i in row_correct]
        while len(unsolved_row_correct) != 0:
            r = unsolved_row_correct[0]
            unsolved_row_correct.pop(0)
            for k in zero_coordinate:
                if r == k[0] and k[1] not in col_correct:
                    col_correct.append(k[1])
                    for k2 in zero_coordinate:
                        if k[1] == k2[1] and k2 in indepent_elements and k2[0] not in row_correct:
                            unsolved_row_correct.append(k2[0])
                            row_correct.append(k2[0])

        line_sum = len(col_correct) + (len(hungarian_array) - len(row_correct))
        if line_sum == n:
            break

        # Step 4
        hungarian_array_temp = hungarian_array + 0
        for i in col_correct:
            hungarian_array_temp[:, i] = np.max(hungarian_array)
        for i in range(len(hungarian_array)):
            if i not in row_correct:
                hungarian_array_temp[i, :] = np.max(hungarian_array)
        min_elements = np.min(hungarian_array_temp)

        for i in range(len(hungarian_array)):
            for j in range(len(hungarian_array[0])):
                if j not in col_correct and i in row_correct:
                    hungarian_array[i, j] -= min_elements
                elif j in col_correct and i not in row_correct:
                    hungarian_array[i, j] += min_elements

    return [i for i in indepent_elements if i[0] < len(input_array) and i[1] < len(input_array[0])]


class SORT(object):
    def __init__(self, max_disappear=30, max_thr=1.0):
        self.track_list = []
        self.score_list = []
        self.max_disappear = max_disappear
        self.max_thr = max_thr

    def update(self, bboxs, scores):
        new_datas = []
        for bbox in bboxs:
            new_datas.append(bbox_2_z(bbox))

        match_array = []
        for track in self.track_list:
            line = []
            for bbox in bboxs:
                if track.pre_current is not None:
                    line.append(get_ciou_loss(np.array(bbox), z_2_bbox(track.pre_current)))
                else:
                    line.append(get_ciou_loss(np.array(bbox), z_2_bbox(track.x_before)))
            match_array.append(line)

        if len(match_array) != 0:
            match_array = np.array(match_array)
            match_result = hungarian(match_array)
        else:
            match_result = []

        not_match_track = list(range(len(self.track_list)))
        not_match_detection = list(new_datas)

        #print(match_array)
        # Matched Tracks
        for i in range(len(match_result)):
            # print(match_array[match_result[i][0], match_result[i][1]])
            if match_array[match_result[i][0], match_result[i][1]] < self.max_thr:
                self.track_list[match_result[i][0]].run(new_datas[match_result[i][1]])
                self.track_list[match_result[i][0]].disappear = 0
                self.track_list[match_result[i][0]].age += 1
                self.score_list[match_result[i][0]] = scores[match_result[i][1]]
                not_match_track.remove(match_result[i][0])
                not_match_detection.remove(new_datas[match_result[i][1]])

        # Unmatched Tracks
        for i in reversed(not_match_track):
            self.track_list[i].disappear += 1
            if self.track_list[i].disappear > self.max_disappear:
                self.track_list.pop(i)
                self.score_list.pop(i)
                continue
            if self.track_list[i].age > 3:
                self.track_list[i].predict(mode=1, replace=True)
                if self.track_list[i].pre_current[2] < 0:  # predict area < 0
                    self.track_list.pop(i)
                    self.score_list.pop(i)

        # Unmatched Detections
        for z in not_match_detection:
            new_tracker = KalmanTracker(initial_position=z)
            self.track_list.append(new_tracker)
            self.score_list.append(0)

        return 1

    def get_bbox_estimate(self):
        bboxs_cor = []
        bboxs_pre = []
        for tracker in self.track_list:
            # Posteriori estimate when a detection match a track
            z_cor = tracker.get_posteriori_estimate()
            if z_cor is not None:
                bbox_cor = z_2_bbox(z_cor)
                bboxs_cor.append(bbox_cor)
            # Priori estimate for every 'adult' track
            z_pre = tracker.get_priori_estimate()
            if z_pre is not None:
                bbox_pre = z_2_bbox(z_pre)
                bboxs_pre.append(bbox_pre)
        return bboxs_cor, bboxs_pre

    # get posteriori estimate or
    def get_output(self):
        output_bboxes = []
        output_scores = []
        for i, tracker in enumerate(self.track_list):
            # Posteriori estimate when a detection match a track
            z_cor = tracker.get_posteriori_estimate()
            if z_cor is not None:
                bbox_cor = z_2_bbox(z_cor)
                output_bboxes.append(bbox_cor)
                output_scores.append(self.score_list[i])
            else:
                z_pre = tracker.get_priori_estimate(t=-1)
                if z_pre is not None:
                    bbox_pre = z_2_bbox(z_pre)
                    output_bboxes.append(bbox_pre)
                    output_scores.append(.0)
        return output_bboxes, output_scores



if __name__ == "__main__":
    import jsonfiler
    seq = jsonfiler.load("seq5.json")
    my_sort = SORT(max_disappear=5)  # 因为是仿真，这里将max_age调小一点

    # A frame consists of several bbox
    for i, bboxs in enumerate(seq):
        # Create white background
        img = cv2.imread("white.jpg")
        save_name = "out/out_%03d.jpg" % i

        # Draw real position
        for bbox in bboxs:
            # Blue rectangle for real position
            cv2.rectangle(img, (bbox[1], bbox[0]), (bbox[3], bbox[2]), (255, 0, 0), 2)

        my_sort.update(bboxs)
        bboxs_cor, bboxs_pre = my_sort.get_bbox_estimate()

        for bbox_cor in bboxs_cor:
            # Green rectangle for posteriori estimate
            cv2.rectangle(img, (bbox_cor[1], bbox_cor[0]), (bbox_cor[3], bbox_cor[2]), (0, 255, 0), 2)

        for bbox_pre in bboxs_pre:
            # Red rectangle for priori estimate
            cv2.rectangle(img, (bbox_pre[1], bbox_pre[0]), (bbox_pre[3], bbox_pre[2]), (0, 0, 255), 2)

        cv2.imwrite(save_name, img)
        print(save_name)
