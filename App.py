import sys
import argparse
import cv2
import numpy as np


def createParser ():
    """
    Парсер для ввода аргумента.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument ('image_url')
 
    return parser


def viewImage(image, name_of_window):
    """
    Просмотр изображения
    """
    cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)
    cv2.imshow(name_of_window, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def openImage(image_url):
    """
    Открытие изображения
    """
    return cv2.imread(image_url)

def overlappingParticles(image):
    """
    Только группы перекрывающихся частиц.
    """
    # эрозия
    kernel = np.array([[0, 0, 1, 0, 0],
                       [0, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 0],
                       [0, 0, 1, 0, 0]], dtype=np.uint8)
    erosion = cv2.erode(image, kernel, iterations=4)

    kernel = np.array([[0, 0, 1, 1, 1, 1, 0, 0],
                       [0, 1, 1, 1, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 1, 1, 1, 0],
                       [0, 0, 1, 1, 1, 1, 0, 0] ], dtype=np.uint8)
    dilation = cv2.dilate(erosion, kernel, iterations=2)
    ret, threshold_image = cv2.threshold(dilation, 30, 255, 0)
    return threshold_image

def singleParticles(originImage, image):
    """
    Только одиночные частицы.
    """
    diff_image = originImage - image
    kernel = np.array([[1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1]], dtype=np.uint8)
    onlySmallPart_image = cv2.morphologyEx(diff_image, cv2.MORPH_OPEN, kernel)

    return onlySmallPart_image

def deleteBorderComponents(image):
    kernel = np.ones((3, 3), np.uint8)
    # удаление лишнего(шумов)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    # выделение границ
    # erosion = cv2.erode(image, kernel, iterations=1)
    erosion = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    image = erosion;

    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(thresh, [max(contours, key=cv2.contourArea)], -1, 255, thickness=-1)
    return image


if __name__ == '__main__':
    #parser = createParser()
    #namespace = parser.parse_args(sys.argv[1:])
    
    image = openImage('./images/circles.jpg')
    onlyBigPart_image = overlappingParticles(image)
    onlySmallPart_image = singleParticles(image, onlyBigPart_image)

    viewImage(deleteBorderComponents(image), 'asd')