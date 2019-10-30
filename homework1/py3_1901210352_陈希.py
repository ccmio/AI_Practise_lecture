from PIL import ImageEnhance
from PIL import ImageOps
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

f = open('./mnist.txt', 'r')
contents = f.readlines()
f.close()

plt.figure(figsize=(12, 6))  # 定义画布大小，方便容纳全部子图
path = './mnist_data_jpg/'
for content in contents:
    value = content.split()
    img_path = path + value[0]
    img = Image.open(img_path)
    img = img.resize((28, 28), Image.ANTIALIAS)
    img = ImageEnhance.Contrast(img).enhance(2)  # 强化一下对比度，显示效果更好
    img = ImageOps.invert(img)  # 反色处理
    img_arr = np.array(img.convert('L'))
    for i in range(28):  # 二值化
        for j in range(28):
            if img_arr[i][j] > 75:  # 试验发现取阈值75效果比较好
                img_arr[i][j] = 255
            else:
                img_arr[i][j] = 0
    p_img = Image.fromarray(img_arr)
    plt.subplot(2, 5, contents.index(content)+1)  # 把处理后的所有子图放在一张画布上展示
    plt.set_cmap('gray')  # 定义plot色域为灰度，不然默认显示的是0为紫色，255为黄色
    plt.axis('off')
    plt.imshow(p_img)
    p_img.save('./已处理 ' + value[1] + '.jpg')  # 处理后的图片存储在根目录
plt.show()


