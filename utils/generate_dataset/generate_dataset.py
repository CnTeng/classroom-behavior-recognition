import pandas as pd
import numpy as np

import cv2 as cv

from modules.face_detection.yunet import YuNet
from modules.pose_estimator.solve_pose import PoseEstimator

backend_target_pairs = [
    [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU],
    [cv.dnn.DNN_BACKEND_CUDA, cv.dnn.DNN_TARGET_CUDA],
    [cv.dnn.DNN_BACKEND_CUDA, cv.dnn.DNN_TARGET_CUDA_FP16],
    [cv.dnn.DNN_BACKEND_TIMVX, cv.dnn.DNN_TARGET_NPU],
    [cv.dnn.DNN_BACKEND_CANN, cv.dnn.DNN_TARGET_NPU],
]


def generate_dataset(behavior_name):
    dataset = pd.read_json("../../dataset/result/alphapose/" + behavior_name + ".json")

    dataset = dataset.drop_duplicates(subset=["image_id"], keep=False)

    assert dataset is not None

    dataset = dataset.reset_index()

    dataset["behavior_index"] = dataset["image_id"].apply(get_behavior_index)
    dataset["face_image"] = dataset["image_id"].apply(get_face_image)
    dataset["body_point"] = dataset["keypoints"].apply(get_body_point)

    return dataset


@staticmethod
def get_behavior_index(image_id):
    keyword = image_id[0]
    if keyword == "d":
        return 0.0
    elif keyword == "l":
        return 1.0
    elif keyword == "p":
        return 2.0
    elif keyword == "t":
        return 3.0
    else:
        return 4.0


@staticmethod
def get_behavior_name(image_id):
    keyword = image_id[0]
    if keyword == "d":
        return "drink"
    elif keyword == "l":
        return "listen"
    elif keyword == "p":
        return "phone"
    elif keyword == "t":
        return "trance"
    else:
        return "write"


def get_face_image(image_id):
    behavior_name = get_behavior_name(image_id)
    backend_id = backend_target_pairs[0][0]
    target_id = backend_target_pairs[0][1]

    model = YuNet(
        modelPath="./modules/face_detection/face_detection_yunet_2022mar.onnx",
        inputSize=[320, 320],
        confThreshold=0.9,
        nmsThreshold=0.3,
        topK=5000,
        backendId=backend_id,
        targetId=target_id,
    )

    image = cv.imread("../../dataset/prev/" + behavior_name + "/" + image_id)

    h, w, _ = image.shape

    # Inference
    model.setInputSize([w, h])

    results = np.array(model.infer(image))
    results = results.flatten()
    len = np.size(results)

    if len > 15:
        mini = 1000000000000000000000
        for i in results.reshape(-1, 15):
            dist = (i[1] + 0.5 * i[3] - 56) ** 2 + (i[0] + 0.5 * i[2] - 56) ** 2
            if dist < mini:
                mini = dist
                i = i.reshape(15).astype(int)

                image = image[
                    i[1] : (i[1] + i[3]),
                    i[0] : (i[0] + i[2]),
                ]

    elif len < 15:
        image = np.zeros((30, 30, 3))
    else:
        results = results.reshape(15).astype(int)
        image = image[
            results[1] : (results[1] + results[3]),
            results[0] : (results[0] + results[2]),
        ]

    if np.sum(image) != 0:
        image = cv.resize(image, (30, 30))
    else:
        image = np.zeros((30, 30, 3))
    image = image.astype(np.uint16)

    ycrcb = cv.cvtColor(image, cv.COLOR_BGR2YCR_CB)
    channels = cv.split(ycrcb)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe.apply(channels[0], channels[0])
    cv.merge(channels, ycrcb)
    cv.cvtColor(ycrcb, cv.COLOR_YCR_CB2BGR, image)
    image = np.transpose(image, (2, 0, 1))

    return image


@staticmethod
def get_body_point(keypoints):
    pose_estimator = PoseEstimator((112, 112))
    keypoints = np.array(keypoints).reshape(136, 3)
    face_point = keypoints[26:94, :2].copy()
    body_point = keypoints[:, :2]
    final_body_point = body_point - body_point[19, :] / 122
    final_body_point[0, :] = body_point[0, :] - body_point[18, :]
    final_body_point[1, :] = body_point[1, :] - body_point[0, :]
    final_body_point[2, :] = body_point[2, :] - body_point[0, :]
    final_body_point[3, :] = body_point[3, :] - body_point[1, :]
    final_body_point[4, :] = body_point[4, :] - body_point[2, :]
    final_body_point[5, :] = body_point[5, :] - body_point[18, :]
    final_body_point[6, :] = body_point[6, :] - body_point[18, :]
    final_body_point[7, :] = body_point[7, :] - body_point[5, :]
    final_body_point[8, :] = body_point[8, :] - body_point[6, :]
    final_body_point[9, :] = body_point[9, :] - body_point[7, :]
    final_body_point[10, :] = body_point[10, :] - body_point[8, :]
    final_body_point[11, :] = body_point[11, :] - body_point[19, :]
    final_body_point[12, :] = body_point[12, :] - body_point[19, :]
    final_body_point[13, :] = body_point[13, :] - body_point[11, :]
    final_body_point[14, :] = body_point[14, :] - body_point[12, :]
    final_body_point[15, :] = body_point[15, :] - body_point[13, :]
    final_body_point[16, :] = body_point[16, :] - body_point[14, :]
    final_body_point[17, :] = body_point[17, :] - body_point[0, :]
    final_body_point[18, :] = body_point[18, :] - body_point[19, :]
    final_body_point[19, :] = body_point[19, :] - body_point[19, :]
    final_body_point[20, :] = body_point[20, :] - body_point[15, :]
    final_body_point[21, :] = body_point[21, :] - body_point[16, :]
    final_body_point[22, :] = body_point[22, :] - body_point[15, :]
    final_body_point[23, :] = body_point[23, :] - body_point[16, :]
    final_body_point[24, :] = body_point[24, :] - body_point[15, :]
    final_body_point[25, :] = body_point[25, :] - body_point[16, :]
    for i in range(26):
        final_body_point[i, :] = final_body_point[i, :] / np.linalg.norm(
            final_body_point[i, :]
        )
    final_body_point[19, :] = [0, 0]
    face_pose = pose_estimator.solve_pose(face_point)
    body_point = np.append(final_body_point, face_pose)
    return body_point


for i in ["drink", "listen", "phone", "trance", "write"]:
    dataset = generate_dataset(i)

    dataset.to_json("../../dataset/result/final/" + i + ".json")

    print(i + " finish")
