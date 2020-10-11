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
    #kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
    kernel = np.array([[0, 0, 1, 0, 0],
                       [0, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 0],
                       [0, 0, 1, 0, 0]], dtype=np.uint8)
    #kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
    erosion = cv2.erode(image, kernel, iterations=4)

    #kernel = np.array([[0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
     #                  [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
      #                 [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
       #                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        #               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         #              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          #             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           #            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            #           [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
             #          [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],], dtype=np.uint8)
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
    viewImage(threshold_image, 'image')


if __name__ == '__main__':
    #parser = createParser()
    #namespace = parser.parse_args(sys.argv[1:])
    
    image = openImage('./images/circles.jpg')
    overlappingParticles(image)
