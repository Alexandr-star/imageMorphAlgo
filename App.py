import sys
import argparse
import cv2

h = 1

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
 
if __name__ == '__main__':
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])
    
    image = openImage(namespace.image_url)

    

    viewImage(image, namespace.image_url)

 
