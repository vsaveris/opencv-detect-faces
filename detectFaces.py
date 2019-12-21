'''
File name: detectFaces.py
    Use of OpenCV's implementation of Haar feature-based cascade classifier, to detect 
    human faces in images.
    The script includes:
        - FacesDetectorCV class: Face detector class based on openCV cascade classifier.
        - Demonstration code.
           
Author: Vasileios Saveris
enail: vsaveris@gmail.com

License: MIT

Date last modified: 21.12.2019

Python Version: 3.7
'''

import cv2

class FacesDetectorCV():
    '''
    FacesDetectorCV object.

    Args:
        classifier_file (string): The path of the openCV Haar feature-based cascade classifier xml file.
            Default value is './opencv_haarcascades/haarcascade_frontalface_alt.xml'
                            
        verbose (boolean): If True logs are enabled (default is False).

    Attributes:
        __classifier_file (string): The path of the openCV Haar feature-based cascade classifier xml file.
            Default value is './opencv_haarcascades/haarcascade_frontalface_alt.xml'

        __faces_cascade_classifier (cv2.CascadeClassifier): The Cascade Classfier created based on the 
            classifier_file.

        __verbose (boolean): If True logs are enabled (default is False).
                                
    Methods:
        detectFaces(): Returns the location of the faces detected in the input image.
    '''
    
    def __init__(self, classifier_file = './opencv_haarcascades/haarcascade_frontalface_alt.xml', verbose = False):
        
        self.__verbose = verbose
        self.__classifier_file = classifier_file
        
        # Load cascade classifier file 
        self.__faces_cascade_classifier = cv2.CascadeClassifier(classifier_file)
        
        if self.__verbose:
            print('Faces Detector initialized.')
            
            
    def detectFaces(self, input_image_file):
        '''
        Returns the location of the faces detected in the input image.

        Args:
            input_image_file (string): The input image file path.

        Raises:
            -

        Returns:
            detected_faces (numpy array): The location of the detected faces in the image.
                Each row is of type (x,y,w,h) where x,y are the coordinates, the w the width
                and h the height of the detected face.
                If an error occures, None is returned.
                If no faces detected, an empty numpy array is returned.
        '''
        
        if self.__verbose:
            print('Detecting faces in image \'', input_image_file, '\'', sep = '')
        
        # Load input image in GRAY scale
        input_image = cv2.imread(input_image_file, cv2.IMREAD_GRAYSCALE)
        
        if input_image is None:
            print('ERROR: Input image \'', input_image_file, '\' not Found.', sep = '')
            return None

        # Detect faces in the gray scaled image, using default parameters of the detectMultiScale
        try:
            detected_faces = self.__faces_cascade_classifier.detectMultiScale(input_image)
        except cv2.error as error:
            print('ERROR: Cascade Classifier \'', self.__classifier_file, '\' not Found.', sep = '')
            return None
               
        if self.__verbose:
            print('Number of faces detected:', len(detected_faces))
        
        return detected_faces


# Demonstration Code
if __name__ == '__main__':
    
    pass