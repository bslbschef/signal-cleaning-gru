import os

import numpy as np


def calculate_sqrt_square_sum(a, b):
    """
    此函数用于计算两个 NumPy 向量 a 和 b 对应元素平方和的平方根组成的向量

    参数:
    a (numpy.ndarray): 第一个输入向量
    b (numpy.ndarray): 第二个输入向量

    返回:
    numpy.ndarray: 计算得到的结果向量
    """
    # 计算平方和向量
    square_sum_vector = a ** 2 + b ** 2
    # 对平方和向量取根号
    result_vector = np.sqrt(square_sum_vector)
    return result_vector

def create_result_folder1(root_path='result'):
    # 检查 result 文件夹是否存在，如果不存在则创建
    result_folder = root_path
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # 获取 result 文件夹下已有的序号文件夹
    existing_folders = [int(f) for f in os.listdir(result_folder) if f.isdigit()]

    # 如果没有已有的序号文件夹，新文件夹序号为 1
    if not existing_folders:
        new_folder_number = 1
    else:
        # 找到最大的序号，并加 1 作为新文件夹的序号
        new_folder_number = max(existing_folders) + 1

    # 新文件夹的完整路径
    new_folder_path = os.path.join(result_folder, str(new_folder_number))

    # 创建新文件夹
    os.makedirs(new_folder_path)

    return new_folder_path

import os


def create_result_folder(root_path='result'):
    # 检查 result 文件夹是否存在，如果不存在则创建
    result_folder = root_path
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # 获取 result 文件夹下已有的序号文件夹
    existing_folders = [int(f) for f in os.listdir(result_folder) if f.isdigit()]

    if existing_folders:
        # 找到最大序号的文件夹
        max_folder_number = max(existing_folders)
        max_folder_path = os.path.join(result_folder, str(max_folder_number))
        # 检查最大序号的文件夹是否为空
        if not os.listdir(max_folder_path):
            # 若为空，直接返回该文件夹路径
            return max_folder_path

    # 如果没有已有的序号文件夹，或者最大序号的文件夹不为空
    # 新文件夹序号为已有最大序号加 1 ，若没有则为 1
    new_folder_number = max(existing_folders) + 1 if existing_folders else 1

    # 新文件夹的完整路径
    new_folder_path = os.path.join(result_folder, str(new_folder_number))

    # 创建新文件夹
    os.makedirs(new_folder_path)

    return new_folder_path


import os

import os


def find_max_numbered_subfolder(folder_path):
    # 检查指定的文件夹是否存在，如果不存在则返回 None
    if not os.path.exists(folder_path):
        return None

    # 存储所有以数字命名的子文件夹及其对应的数字序号
    numbered_folders = {}
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        # 检查是否为文件夹且名称为纯数字
        if os.path.isdir(item_path) and item.isdigit():
            numbered_folders[int(item)] = item_path

    # 如果没有找到以数字命名的子文件夹，返回 None
    if not numbered_folders:
        return None

    # 找到最大的数字序号
    max_number = max(numbered_folders.keys())
    # 返回最大序号对应的子文件夹的完整路径
    return numbered_folders[max_number]