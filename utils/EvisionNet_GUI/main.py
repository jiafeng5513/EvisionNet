# -*- coding:UTF-8 -*-
import sys
import EvisionNet_GUI  # TODO: Change according to the file name of the main window
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog,QFileDialog,QMessageBox


class EvisionNetGUI(QMainWindow):
    def __init__(self):
        super(EvisionNetGUI, self).__init__()
        self.ui = EvisionNet_GUI.Ui_MainWindow()
        self.ui.setupUi(self)
        # signal and solts
        self.signals_handler()

    def signals_handler(self):
        self.ui.pushButton_brewse.clicked.connect(self.OnPushed_broswer)
        pass

    def OnPushed_broswer(self):
        #print("get~")
        #QMessageBox.information(self, 'msg', 'hello', QMessageBox.Yes | QMessageBox.No)
        fname = QFileDialog.getOpenFileName(self, 'open file', './')
        if fname[0]:
            try:
                f = open(fname[0], 'r')
                with f:
                    data = f.read()
                    self.textEdit.setText(data)
            except:
                self.textEdit.setText("NOOOOO!")
        pass

if __name__ == '__main__':
    app = QApplication(sys.argv)  # pyqt appliaction object
    Window = EvisionNetGUI()  # TODO: Change according to the type of main window
    Window.show()
    sys.exit(app.exec_())