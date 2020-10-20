import sys
import argparse
import cv2
import numpy as np

color = (255, 255, 255)

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
    """
    Только частицы касающиеся краев изображения
    """
    ret, binary_image = cv2.threshold(image, 127, 255, 0)
    save(binary_image, "binary_image")

    big_kernel = np.ones((5, 5), np.uint8)
    withoutNoise_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, big_kernel)
    save(withoutNoise_image, "withoutNoise_image")

    kernel = np.ones((3, 3), np.uint8)
    contours_image = cv2.morphologyEx(withoutNoise_image, cv2.MORPH_GRADIENT, kernel)
    save(contours_image, "contours_image")

    imageBGR2GRAY = cv2.cvtColor(contours_image, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(imageBGR2GRAY, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    filledContours = np.zeros((contours_image.shape[0], contours_image.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        cv2.drawContours(filledContours, contours, int(i), color, thickness=-1)
    save(filledContours, "filledContours")

    withoutBorder_image = cv2.morphologyEx(filledContours, cv2.MORPH_OPEN, big_kernel)
    save(withoutBorder_image, "withoutBorder_image")

    onlyBorderPart_image = singleParticles(image, withoutBorder_image)
    save(withoutBorder_image, "onlyBorderPart_image")

    return onlyBorderPart_image

def save(image, name):
    url = './savedimage/{0}.jpg'.format(name)
    cv2.imwrite(url, image)


if __name__ == '__main__':
    #parser = createParser()
    #namespace = parser.parse_args(sys.argv[1:])
    
    image = openImage('./images/circles.jpg')

    
    onlyBigPart_image = overlappingParticles(image)
    onlySmallPart_image = singleParticles(image, onlyBigPart_image)
    onlyBorderPart_image = deleteBorderComponents(image)
    viewImage(onlyBorderPart_image, 'asd')