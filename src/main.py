import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# 1. 加载MNIST数据集（这里使用sklearn自带的类似MNIST数据集）
digits = load_digits()
images = digits.images
labels = digits.target

# 2. 提取HoG特征
hog_features = []
for image in images:
    # 调整pixels_per_cell参数以适应小尺寸图像
    fd = hog(image, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(1, 1))
    hog_features.append(fd)
hog_features = np.array(hog_features)

# 3. 划分数据集
X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(
    hog_features, labels, images, test_size=0.3, random_state=42
)

# 4. 训练SVM分类器
clf = SVC(C=1.0, kernel="rbf", gamma="scale")
clf.fit(X_train, y_train)

# 5. 评估模型
accuracy = clf.score(X_test, y_test)
print(f"模型准确率: {accuracy}")

# 6. 进行预测并输出结果
predictions = clf.predict(X_test)

# 7. 可视化原始图像和预测结果
num_images = 10  # 显示前10张图像和预测结果
fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
for i in range(num_images):
    axes[i].imshow(images_test[i], cmap="gray")
    axes[i].set_title(f"Pred: {predictions[i]}")
    axes[i].axis("off")

plt.show()
