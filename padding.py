from PIL import Image, ImageOps
import numpy as np
import cv2
from matplotlib import pyplot as plt

def padding(img, expected_size):
    desired_size = expected_size
    delta_width = desired_size[0] - img.size[0]
    delta_height = desired_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)


def resize_with_padding(img, expected_size):
    img.thumbnail((expected_size[0], expected_size[1]))
    # print(img.size)
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding, fill = (234, 238, 239))

def get_cloth_mask(image_path):

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(image)
    cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)

    return mask

if __name__ == "__main__":
    
    img = Image.open("D:/Datathon/PF-AFN_root/PF-AFN_test/dataset/test_img/test5.jpg")
    img = img.crop((140,10,460,600)) # Human
    #img = img.crop((90,0,510,600)) # Cloth
    #img = img.convert('RGB')
    #print(img.getpixel((1, 1)))
    img = resize_with_padding(img, (192, 256))
    img.show()
    #print(img.size)
    img.save("D:/Datathon/PF-AFN_root/PF-AFN_test/dataset/test_img/18_0.jpg")
    '''
    mask = get_cloth_mask("D:/Datathon/PF-AFN_root/PF-AFN_test/dataset/test_clothes/12_1.jpg")
    PATH_SAVE = f"D:/Datathon/PF-AFN_root/PF-AFN_test/dataset/test_edge/{12}_1.jpg"
    cv2.imwrite(PATH_SAVE, cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))
    '''