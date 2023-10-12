import cv2
import numpy as np


def rotate_image_and_keypoints(image, keypoints, angle):
    keypoints = keypoints.reshape((-1, 2)).astype(np.float).astype(np.int)
    # 计算旋转中心点
    center = (image.shape[1] // 2, image.shape[0] // 2)

    # 定义旋转矩阵
    theta = np.radians(angle)
    rotation_matrix = np.array(
        [[np.cos(theta), -np.sin(theta), center[0] * (1 - np.cos(theta)) + center[1] * np.sin(theta)],
         [np.sin(theta), np.cos(theta), center[1] * (1 - np.cos(theta)) - center[0] * np.sin(theta)],
         [0, 0, 1]])

    # 旋转图像
    rotated_image = cv2.warpAffine(image, rotation_matrix[:2, :], (image.shape[1], image.shape[0]))

    # 扩展关键点坐标为齐次坐标
    homogeneous_keypoints = np.concatenate([keypoints, np.ones((keypoints.shape[0], 1))], axis=1)

    # 旋转关键点坐标
    rotated_keypoints = np.dot(rotation_matrix, homogeneous_keypoints.T).T[:, :2]
    return rotated_image, rotated_keypoints.reshape(-1)

def rotate_image_and_landmarks(image, face_landmarks, angle_degrees, center):
    """
    旋转图像和人脸关键点

    参数:
    - image: 要旋转的图像
    - face_landmarks: 一个形状为 (N, 2) 的 NumPy 数组，表示 N 个人脸关键点的坐标
    - angle_degrees: 旋转角度（以度为单位）
    - center: 旋转中心的坐标 (x, y)

    返回值:
    旋转后的图像和人脸关键点元组 (rotated_image, rotated_landmarks)
    """

    # 获取图像的尺寸
    height, width = image.shape[:2]

    # 将角度转换为弧度
    angle_rad = np.deg2rad(angle_degrees)

    # 构建旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)

    # 旋转图像
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    # 将关键点转换为浮点型数组
    face_landmarks = np.array(face_landmarks, dtype=np.float32)

    # 添加齐次坐标（第三个维度为1）
    ones = np.ones((face_landmarks.shape[0], 1))
    face_landmarks_homogeneous = np.hstack((face_landmarks, ones))

    # 应用旋转矩阵到关键点
    rotated_landmarks_homogeneous = np.dot(face_landmarks_homogeneous, rotation_matrix.T)

    # 移除齐次坐标（第三个维度）
    rotated_landmarks = rotated_landmarks_homogeneous[:, :2]

    # 计算旋转后的左上角和右下角关键点坐标
    x_min = np.min(rotated_landmarks[:, 0])
    y_min = np.min(rotated_landmarks[:, 1])
    x_max = np.max(rotated_landmarks[:, 0])
    y_max = np.max(rotated_landmarks[:, 1])

    # 修正旋转后的左上角和右下角关键点坐标，确保仍然在脸部区域内
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(width - 1, x_max)
    y_max = min(height - 1, y_max)

    # 裁剪旋转后的图像
    cropped_image = rotated_image[int(y_min):int(y_max)+1, int(x_min):int(x_max)+1]

    # 调整关键点坐标偏移量
    rotated_landmarks[:, 0] -= x_min
    rotated_landmarks[:, 1] -= y_min

    return cropped_image, rotated_landmarks


def flip_image_and_keypoints(image, keypoints):
    keypoints = keypoints.reshape((-1, 2)).astype(np.float).astype(np.int)

    # 水平翻转图像
    flipped_image = cv2.flip(image, 1)

    # 翻转关键点坐标
    flipped_keypoints = keypoints.copy()
    flipped_keypoints[:, 0] = image.shape[1] - keypoints[:, 0]

    return flipped_image, flipped_keypoints.reshape(-1)
