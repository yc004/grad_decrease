"""
用于保存手写数字数据集的图像和标签
"""

import numpy as np
from sklearn.datasets import load_digits

digits = load_digits()
images = digits.images
labels = digits.target


np.save("handwritten_images.npy", images)
np.save("handwritten_labels.npy", labels)
