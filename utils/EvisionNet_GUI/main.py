# -*- coding:UTF-8 -*-
import sys
import EvisionNet_GUI  # TODO: Change according to the file name of the main window
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog

if __name__ == '__main__':
    app = QApplication(sys.argv)  # pyqt appliaction object
    Window = QMainWindow()  # TODO: Change according to the type of main window
    ui = EvisionNet_GUI.Ui_MainWindow()
    ui.setupUi(Window)
    Window.show()
    sys.exit(app.exec_())