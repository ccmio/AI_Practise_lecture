from PIL import ImageEnhance
from PIL import ImageOps
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

f = open('./mnist.txt', 'r')
contents = f.readlines()
f.close()

plt.figure(figsize=(12, 6))
path = './mnist_data_jpg/'
for content in contents:
    value = content.split()
    img_path = path + value[0]
    img = Image.open(img_path)
    img = img.resize((28, 28), Image.ANTIALIAS)
    img = ImageEnhance.Contrast(img).enhance(2)
    img = ImageOps.invert(img)
    img_arr = np.array(img.convert('L'))
    for i in range(28):
        for j in range(28):
            if img_arr[i][j] > 75:
                img_arr[i][j] = 255
            else:
                img_arr[i][j] = 0
    p_img = Image.fromarray(img_arr)
    plt.subplot(2, 5, contents.index(content)+1)
    plt.set_cmap('gray')
    plt.axis('off')
    plt.imshow(p_img)
    p_img.save('./已处理 ' + value[1] + '.jpg')
plt.show()


