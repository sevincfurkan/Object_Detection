import sys
from PyQt5 import QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage
import cv2

import design
from image import yolo3_image
from video import yolo3_video
from camera import yolo3_camera

class MainApp(QtWidgets.QMainWindow, design.Ui_MainWindow):
    
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.pushButton.clicked.connect(self.image_label_object)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.viewCam)
        self.pushButton_3.clicked.connect(self.controlTimer)
        
        self.pushButton_2.clicked.connect(self.label_video)
        
    
    def label_video(self):
        video_path = \
            QtWidgets.QFileDialog.getOpenFileName(self, 'Choose Image to Open',
                                                  '.',
                                                  '*.mp4')
        print(type(video_path))
        print(video_path[0])
        print(video_path[1])
        
        video_path = video_path[0]
        yolo3_video(video_path)
    def viewCam(self):
        ret, image = self.cap.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channel = image.shape
        step = channel * width
        qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qImg))
    
    def controlTimer(self):
        if not self.timer.isActive():
            self.pushButton_3.setText("RECAMERA")
            yolo3_camera()
        else:
            self.timer.stop()
            self.cap.release()
            self.pushButton_3.setText("CAMERA")
        
    def image_label_object(self):
        image_path = \
            QtWidgets.QFileDialog.getOpenFileName(self, 'Choose Image to Open',
                                                  '.',
                                                  '*.png *.jpg *.bmp')
        print(type(image_path))
        print(image_path[0])
        print(image_path[1])

        image_path = image_path[0]
        yolo3_image(image_path)
    
def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainApp()
    window.show()
    app.exec_()
    
if __name__ == '__main__':
    main()
