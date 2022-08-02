import sys
import os
import math
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QLabel, QFileDialog, QTableWidget, QAction, QTableWidgetItem, QHeaderView, QMenu, QMenuBar
from PyQt5.QtGui import QPixmap
from PyQt5 import uic, QtWidgets
import pandas as pd
from processing import Ui_PreProcess
from invert_main import Ui_Invert_Window
from howtouse import Ui_HowToUse
from WhatisCRUISE import Ui_WhatisCRUISE
from errorwindow import Ui_ErrorWindow
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from tensorflow.keras.models import load_model
import cv2

class UI(QMainWindow):
    def __init__(self):
        super(UI, self).__init__()

        #Load the UI file
        uic.loadUi("mainwindow.ui", self)

        #Define Widgets
        self.button = self.findChild(QPushButton, "pushButton")
        self.label = self.findChild(QLabel, "label_3")
        self.button2 = self.findChild(QPushButton, "pushButton_2")
        self.button3 = self.findChild(QPushButton, "pushButton_3")

        #Click Import Data Button
        self.button.clicked.connect(self.import_file)

        #Click Process and Invert Button
        self.button2.clicked.connect(self.error1)

        #Click Invert Button
        self.button3.clicked.connect(self.error2)


        #Show the App
        self.show()

    #Function when the Import Data Button is clicked
    def import_file(self):
        # self.label.setText("You Clicked the Button")
        global dir
        global fp
        fname = QFileDialog.getOpenFileName(self, "Import Data","","All Files (*);;Excel Files (*.xlsx);;CSV Files (*.csv);;DAT Files (.dat)")
        fp = self.label.text()
        # Display Filename
        if fname:
            self.label.setText(fname[0])

        dir = fname[0]

    #Function when the Process and Invert Button is clicked
    #This needs to be changed to reading a .dat file when raw data is already available
    def read(self):
        # data = pd.read_excel(r'/Users/markjeromesanpedro/Documents/CRUISE/Sample Dataset.xlsx')
        global data
        global directory
        directory = dir
        data = pd.read_excel(directory)
        print(data[['Latitude', 'Longitude', 'Elevation','Voltage','Current','Phase Shift']])
        self.close()

    #Function when the Invert Button is clicked
    def read1(self):
        # data = pd.read_excel(r'/Users/markjeromesanpedro/Documents/CRUISE/Sample Dataset.xlsx')
        global data
        global directory
        global table
        global f
        global dataPATH
        directory = dir

        dataPATH = directory

        f = open(directory)
        lines: list[str] = f.readlines()
        x_list = []
        s_list = []
        R_list = []

        length_data = int((len(lines) - 11 + 60) / 6)
        data = np.zeros((6, length_data))
        for dataline in lines[7:-4]:
            datapoint = dataline.replace('\n', '')
            datapoint = np.fromstring(datapoint, dtype=float, sep=' ')
            x_loc = int((datapoint[0] - 0.5) / 0.25)
            depth = int(datapoint[1] - 1)
            data[depth, x_loc] = datapoint[2]
            x_list.append(datapoint[0])
            s_list.append(datapoint[1])
            R_list.append(datapoint[2])
        table = pd.DataFrame(list(zip(x_list, s_list, R_list)), columns=['x-midpoint', 's', 'R'])
        print(table)

        self.close()


    #Function when the Import File Button is clicked (Process and Invert Window)
    def import_read(self):
        global data
        global directory
        directory = dir
        self.ui.filepath.setText(directory)
        if directory == '':
            self.error.show()
        else:
            self.ui.filepath.setText(directory)
            data = pd.read_excel(directory)
            print(data[['Latitude', 'Longitude', 'Elevation', 'Voltage', 'Current', 'Phase Shift']])
            self.displayData()
            self.Pro_Data()

    # Function when the Import File Button is clicked (Invert Window)
    def import_read1(self):
        global data
        global directory
        global table
        global f
        global dataPATH
        directory = dir
        self.ui2.filepath.setText(directory)
        if directory == '':
            self.error.show()
        else:
            self.ui2.filepath.setText(directory)
            dataPATH = directory
            f = open(directory)
            lines: list[str] = f.readlines()
            x_list = []
            s_list = []
            R_list = []

            length_data = int((len(lines) - 11 + 60) / 6)
            data = np.zeros((6, length_data))
            for dataline in lines[7:-4]:
                datapoint = dataline.replace('\n', '')
                datapoint = np.fromstring(datapoint, dtype=float, sep=' ')
                x_loc = int((datapoint[0] - 0.5) / 0.25)
                depth = int(datapoint[1] - 1)
                data[depth, x_loc] = datapoint[2]
                x_list.append(datapoint[0])
                s_list.append(datapoint[1])
                R_list.append(datapoint[2])
            table = pd.DataFrame(list(zip(x_list, s_list, R_list)), columns=['x-midpoint', 's', 'R'])
            print(table)
            self.dispInv()

    #Open the Processing Window
    def openProcess(self):
        self.window = QtWidgets.QMainWindow()
        self.ui = Ui_PreProcess()
        self.ui.setupUi(self.window)
        self.ui.filepath.setText(directory)
        self.window.show()

    #Open the Invert Window
    def openInvert(self):
        self.invert = QtWidgets.QMainWindow()
        self.ui2 = Ui_Invert_Window()
        self.ui2.setupUi(self.invert)
        self.ui2.filepath.setText(directory)
        self.invert.show()

    #Open How to Use Window
    def howTo(self):
        self.how = QtWidgets.QMainWindow()
        self.ui3 = Ui_HowToUse()
        self.ui3.setupUi(self.how)
        self.how.show()

    #Open What is CRUISE Window
    def whatIs(self):
        self.what = QtWidgets.QMainWindow()
        self.ui4 = Ui_WhatisCRUISE()
        self.ui4.setupUi(self.what)
        self.what.show()

    #Open Error Window for Process and Invert Button
    def error1(self):
        self.error = QtWidgets.QMainWindow()
        self.ui5 = Ui_ErrorWindow()
        self.ui5.setupUi(self.error)
        fp = self.label.text()

        if fp == 'Directory/Filename' or fp == '':
            self.error.show()
        else:
            self.read()
            self.openProcess()
            self.displayData()
            self.Pro_Data()
            self.menu()
            self.error.close()

    #Open Error Window for Invert Button
    def error2(self):
        self.error = QtWidgets.QMainWindow()
        self.ui5 = Ui_ErrorWindow()
        self.ui5.setupUi(self.error)
        fp = self.label.text()

        if fp == 'Directory/Filename' or fp == '':
            self.error.show()
        else:
            self.read1()
            self.openInvert()
            self.dispInv()
            self.menu1()
            self.error.close()

    #Display Data in Processing Window
    def displayData(self):
        global final_table
        ex_dir = self.ui.filepath.text()
        load_data = pd.read_excel(ex_dir)

        if load_data.size == 0:
            return
        load_data.fillna('', inplace=True)
        self.ui.RawData.setRowCount(load_data.shape[0])
        self.ui.RawData.setColumnCount(load_data.shape[1])
        self.ui.RawData.setHorizontalHeaderLabels(load_data.columns)

        # Writing the data into the RawData Table
        for row in load_data.iterrows():
            values = row[1]
            for col_index, value in enumerate(values):
                if isinstance(value, (float, int)):
                    value = '{0:0}'.format(value)
                tableItem = QTableWidgetItem(str(value))
                self.ui.RawData.setItem(row[0], col_index, tableItem)

            # Converting Latitude, Longitude to Cartesian Coordinates
            global final_table
            x_list = []
            y_list = []
            ordered_pair = []

            def LatLngToCart():
                for i in range(len(data)):
                    lat = math.radians(data.at[i, 'Latitude'])
                    long = math.radians(data.at[i, 'Longitude'])
                    h = data.at[i, 'Elevation']
                    # h=0
                    a = 6378137  # WGS 84 Major axis
                    b = 6356752.3142  # WGS 84 Minor axis
                    e2 = 1 - (b ** 2 / a ** 2)
                    N = float(a / math.sqrt(1 - e2 * (math.sin(abs(lat)) ** 2)))
                    x_coor = (N + h) * math.cos(lat) * math.cos(long)
                    y_coor = (N + h) * math.cos(lat) * math.sin(long)
                    x_list.append(x_coor)
                    y_list.append(y_coor)
                    ordered_pair.append((x_coor, y_coor))
            LatLngToCart()

            # Normalizing the data
            xmin = min(x_list)
            xmax = max(x_list)
            ymax = max(y_list)
            ymin = min(y_list)

            for i, x in enumerate(x_list):
                x_list[i] = (x - xmin) / (xmax - xmin)

            for i, y in enumerate(y_list):
                y_list[i] = (y - ymin) / (ymax - ymin)

        initial_table = pd.DataFrame(list(zip(x_list, y_list)), columns=['x', 'y'])
        print(initial_table)

        # Getting the x-midpoint
        dx = [x_list[i] + x_list[i - 1] for i in range(1, len(x_list))]
        dy = [y_list[i] + y_list[i - 1] for i in range(1, len(y_list))]
        x_midpoint = []
        y_midpoint = []
        dipole_spacing = []

        for x in dx:
            def res(x): return x / 2

            ds = 1
            x_midpoint.append(res(x))
            dipole_spacing.append(ds)

        for y in dy:
            def res1(y): return y / 2

            y_midpoint.append(res1(y))

        final_table = pd.DataFrame(list(zip(x_midpoint, dipole_spacing)), columns=['x-midpoint', 's'])
        final_table['R'] = (data['Voltage'] / data['Current']) * ((math.pi) / ((1 / 1) - (1 / math.sqrt(2))))
        print(final_table)

    def Pro_Data(self):
        if final_table.size == 0:
            return
        final_table.fillna('', inplace=True)
        self.ui.ProcessedData.setRowCount(final_table.shape[0])
        self.ui.ProcessedData.setColumnCount(final_table.shape[1])
        self.ui.ProcessedData.setHorizontalHeaderLabels(final_table.columns)

        # Writing the data into the ProcessedData Table
        for row in final_table.iterrows():
            values = row[1]
            for col_index1, value1 in enumerate(values):
                if isinstance(value1, (float, int)):
                    value1 = '{0:0}'.format(value1)
                tableItem2 = QTableWidgetItem(str(value1))
                self.ui.ProcessedData.setItem(row[0], col_index1, tableItem2)

    #Quasi 2D Map Generation
    def preMapQ(self, readPATH, half_width):
        f1 = open(readPATH, 'r')
        lines: list[str] = f1.readlines()
        length_data = int((len(lines) - 11 + 60) / 6)
        data = np.zeros((6, length_data))
        for dataline in lines[7:-4]:
            datapoint = dataline.replace('\n', '')
            datapoint = np.fromstring(datapoint, dtype=float, sep=' ')
            x_loc = int((datapoint[0] - 0.5) / 0.25)
            depth = int(datapoint[1] - 1)
            data[depth, x_loc] = datapoint[2]

        if half_width == 0:
            data = data.transpose()
        else:
            data_map = data.transpose()
            expanded_data_map = np.empty((len(data_map), 1))

            for j in range(-(half_width), half_width + 1):
                if j < 0:
                    feature = np.pad(data_map, ((-j, 0), (0, 0)), mode='constant')[:j, :]
                elif j > 0:
                    feature = np.pad(data_map, ((0, j), (0, 0)), mode='constant')[j:, :]
                else:
                    feature = data_map

                expanded_data_map = np.append(expanded_data_map, feature, axis=1)
                data = expanded_data_map[:, 1:]

        return data

    def generate_qmap(self, dataPATH, modelPATH, half_width):

        cols = 6
        rows = 5

        data = self.preMapQ(dataPATH, half_width)

        data_scaled = np.log(1 + data) / np.log(100)
        data_scaled = np.reshape(data_scaled, (len(data_scaled), rows, cols))
        data_scaled = np.transpose(data_scaled, (0, 2, 1))
        data_scaled = np.expand_dims(data_scaled, axis=3)

        model = load_model(modelPATH)
        preds_scaled = model.predict(data_scaled)

        preds = np.power(100, preds_scaled)
        preds = preds.transpose()
        preds[preds <= 0] = 0.01

        x = np.linspace(0.25, 100, 397)
        z = np.linspace(-0.1, -3, 30)

        X, Z = np.meshgrid(x, z)

        fig = plt.figure(figsize=(13, 6))

        cp = plt.contourf(X, Z, preds, locator=ticker.LogLocator())

        plt.colorbar(cp)

        # plt.title('Quasi-2D Map', fontweight='bold', fontsize=28)
        plt.xlabel('x (m)', fontweight='bold', fontsize=18)
        plt.ylabel('z (m)', fontweight='bold', fontsize=18)

        plt.savefig('./quasi2d.png')
        os.remove('./quasi2d.png')
        plt.savefig('./quasi2d.png')
        plt.close()

        # return preds

    def view_qmap(self, dataPATH, modelPATH, half_width):

        cols = 6
        rows = 5

        data = self.preMapQ(dataPATH, half_width)

        data_scaled = np.log(1 + data) / np.log(100)
        data_scaled = np.reshape(data_scaled, (len(data_scaled), rows, cols))
        data_scaled = np.transpose(data_scaled, (0, 2, 1))
        data_scaled = np.expand_dims(data_scaled, axis=3)

        model = load_model(modelPATH)
        preds_scaled = model.predict(data_scaled)

        preds = np.power(100, preds_scaled)
        preds = preds.transpose()
        preds[preds <= 0] = 0.01

        x = np.linspace(0.25, 100, 397)
        z = np.linspace(-0.1, -3, 30)

        X, Z = np.meshgrid(x, z)

        fig = plt.figure(figsize=(20, 6))

        cp = plt.contourf(X, Z, preds, locator=ticker.LogLocator())

        plt.colorbar(cp)

        plt.title('Quasi-2D Map', fontweight='bold', fontsize=28)
        plt.xlabel('x (m)', fontweight='bold', fontsize=18)
        plt.ylabel('z (m)', fontweight='bold', fontsize=18)

        plt.show()

        return preds

    #2D Map Generation

    def preMap2D(self, readPATH):
        f = open(readPATH, 'r')
        lines = f.readlines()

        length_data = int((len(lines) - 11 + 60) / 6)
        data = np.zeros((6, length_data))

        for dataline in lines[7:-4]:
            datapoint = dataline.replace('\n', '')
            datapoint = np.fromstring(datapoint, dtype=float, sep=' ')
            x_loc = int((datapoint[0] - 0.5) / 0.25)
            depth = int(datapoint[1] - 1)
            data[depth, x_loc] = datapoint[2]

        return data

    def generate_2D(self, dataPATH, modelPATH ):

        data = self.preMap2D(dataPATH)

        data_scaled = np.log(1 + data) / np.log(100)

        data_scaled = cv2.resize(data_scaled, dsize=(800, 32), interpolation=cv2.INTER_LINEAR)
        data_scaled = np.reshape(data_scaled, (1, 32, 800))
        data_scaled = np.expand_dims(data_scaled, axis=3)

        model = load_model(modelPATH)
        preds_scaled = model.predict(data_scaled)
        preds = np.power(100, preds_scaled)
        preds = np.reshape(preds, (32, 800))

        preds[preds <= 0] = 0.01

        x = np.linspace(0.25, 100, 800)
        z = np.linspace(-0.1, -3, 32)

        X, Z = np.meshgrid(x, z)

        fig = plt.figure(figsize=(13, 6))

        cp = plt.contourf(X, Z, preds, locator=ticker.LogLocator())
        # cp = plt.contourf(X, Z, preds)

        cbar = plt.colorbar(cp)

        # plt.title('2D Inversion Map')
        plt.xlabel('x (m)', fontweight='bold', fontsize=18)
        plt.ylabel('z (m)', fontweight='bold', fontsize=18)

        # cbar.set_label('Resistivity', rotation=270)

        plt.savefig('./2d.png')
        os.remove('./2d.png')
        plt.savefig('./2d.png')
        plt.close()

    def view_2D(self, dataPATH, modelPATH):

        data = self.preMap2D(dataPATH)

        data_scaled = np.log(1 + data) / np.log(100)

        data_scaled = cv2.resize(data_scaled, dsize=(800, 32), interpolation=cv2.INTER_LINEAR)
        data_scaled = np.reshape(data_scaled, (1, 32, 800))
        data_scaled = np.expand_dims(data_scaled, axis=3)

        model = load_model(modelPATH)
        preds_scaled = model.predict(data_scaled)
        preds = np.power(100, preds_scaled)
        preds = np.reshape(preds, (32, 800))

        preds[preds <= 0] = 0.01

        x = np.linspace(0.25, 100, 800)
        z = np.linspace(-0.1, -3, 32)

        X, Z = np.meshgrid(x, z)

        fig = plt.figure(figsize=(20, 6))

        cp = plt.contourf(X, Z, preds, locator=ticker.LogLocator())
        # cp = plt.contourf(X, Z, preds)

        cbar = plt.colorbar(cp)

        plt.title('2D Inversion Map', fontweight='bold', fontsize=28)
        plt.xlabel('x (m)', fontweight='bold', fontsize=18)
        plt.ylabel('z (m)', fontweight='bold', fontsize=18)

        # cbar.set_label('Resistivity', rotation=270)

        plt.show()


    #Display Data in Invert Window
    def dispInv(self):
        if table.size == 0:
            return
        table.fillna('', inplace=True)
        self.ui2.ProcessedData.setRowCount(table.shape[0])
        self.ui2.ProcessedData.setColumnCount(table.shape[1])
        self.ui2.ProcessedData.setHorizontalHeaderLabels(table.columns)

        # Writing the data into the ProcessedData Table
        for row in table.iterrows():
             values = row[1]
             for col_index, value in enumerate(values):
                 if isinstance(value, (float, int)):
                     value = '{0:0}'.format(value)
                 tableItem = QTableWidgetItem(str(value))
                 self.ui2.ProcessedData.setItem(row[0], col_index, tableItem)

    #Process and Invert Menu Bar and Widgets
    def menu(self):
        self.ui.actionMain_Menu.triggered.connect(self.window.close)
        self.ui.actionMain_Menu.triggered.connect(self.__init__)
        self.ui.actionImport_File.triggered.connect(self.import_file)
        self.ui.actionImport_File.triggered.connect(self.import_read)
        self.ui.actionHow_to_use.triggered.connect(self.howTo)
        self.ui.actionWhat_is_Cruise.triggered.connect(self.whatIs)
        self.ui.SelectMap1.currentTextChanged.connect(self.combo1)
        self.ui.SelectMap1.currentTextChanged.connect(self.combo2)
        self.ui.SelectMap2.currentTextChanged.connect(self.combo1)
        self.ui.SelectMap2.currentTextChanged.connect(self.combo3)

    def combo1(self):
        global type
        global type1
        type = self.ui.SelectMap1.currentText()
        type1 = self.ui.SelectMap2.currentText()
        self.ui.Map1Label.setText(type)
        self.ui.Map2Label.setText(type1)

    def combo2(self):
        if type == 'Quasi 2D':
            print(type)
        if type == '2D Map':
            print(type)
        if type == '3D Map':
            print(type)
        if type == 'Select Map Type':
            self.ui.Map1Label.setText("Map 1")

    def combo3(self):
        if type1 == 'Quasi 2D':
            print(type1)

        if type1 == '2D Map':
            print(type1)
        if type == '3D Map':
            print(type1)
        if type1 == 'Select Map Type':
            self.ui.Map2Label.setText("Map 2")

    #Invert Menu Bar and Widgets
    def menu1(self):
        self.ui2.actionMain_Menu.triggered.connect(self.invert.close)
        self.ui2.actionMain_Menu.triggered.connect(self.__init__)
        self.ui2.actionImport_File.triggered.connect(self.import_file)
        self.ui2.actionImport_File.triggered.connect(self.import_read1)
        self.ui2.actionHow_to_use.triggered.connect(self.howTo)
        self.ui2.actionWhat_is_Cruise.triggered.connect(self.whatIs)
        self.ui2.SelectMap1.activated.connect(self.combo4)
        self.ui2.SelectMap1.currentTextChanged.connect(self.combo4)
        self.ui2.Generate1.clicked.connect(self.combo5)
        self.ui2.ViewMap1.clicked.connect(self.view1)
        self.ui2.SelectMap2.activated.connect(self.combo4)
        self.ui2.SelectMap1.currentTextChanged.connect(self.combo4)
        self.ui2.Generate2.clicked.connect(self.combo6)
        self.ui2.VIewMap2.clicked.connect(self.view2)
        # self.ui2.SelectMap2.currentTextChanged.connect(self.combo6)

    #Combo Box

    def combo4(self):
         global type2
         global type3
         type2 = self.ui2.SelectMap1.currentText()
         type3 = self.ui2.SelectMap2.currentText()

    #Select Map 1
    def combo5(self):
        if type2 == 'Quasi 2D':
            print(type2)
            modelPATH = r'./quasi2d.h5'
            self.generate_qmap(dataPATH, modelPATH, 2)
            pixmap = QPixmap('./quasi2d.png')
            self.ui2.Map1Holder.setPixmap(pixmap)
            self.ui2.Map1Label.setText(type2)
        if type2 == '2D Map':
            print(type2)
            modelPATH = r'./2d.h5'
            self.generate_2D(dataPATH, modelPATH)
            pixmap2 = QPixmap('./2d.png')
            self.ui2.Map1Holder.setPixmap(pixmap2)
            self.ui2.Map1Label.setText(type2)
        if type2 == '3D Map':
            print(type2)
            self.ui2.Map1Label.setText(type2)
        if type2 == 'Select Map Type':
            self.ui2.Map1Label.setText("Map 1")

    #Select Map 2
    def combo6(self):
        if type3 == 'Quasi 2D':
            print(type3)
            modelPATH = r'./quasi2d.h5'
            self.generate_qmap(dataPATH, modelPATH, 2)
            pixmap = QPixmap('./quasi2d.png')
            self.ui2.Map1Holder_2.setPixmap(pixmap)
            self.ui2.Map2Label.setText(type3)
        if type3 == '2D Map':
            print(type3)
            modelPATH = r'./2d.h5'
            self.generate_2D(dataPATH, modelPATH)
            pixmap2 = QPixmap('./2d.png')
            self.ui2.Map1Holder_2.setPixmap(pixmap2)
            self.ui2.Map2Label.setText(type2)
            self.ui2.Map2Label.setText(type3)
        if type3 == '3D Map':
            print(type3)
            self.ui2.Map2Label.setText(type3)
        if type3 == 'Select Map Type':
            self.ui2.Map2Label.setText("Map 2")


    #View Map
    def view1(self):
        type4 = self.ui2.Map1Label.text()
        if type4 == 'Quasi 2D':
            print(type4)
            modelPATH = r'./quasi2d.h5'
            self.view_qmap(dataPATH, modelPATH, 2)
        if type4 == '2D Map':
            print(type4)
            modelPATH = r'./2d.h5'
            self.view_2D(dataPATH, modelPATH)
        if type4 == '3D Map':
            print(type4)
        if type4 == 'Select Map Type':
            self.ui2.Map1Label.setText("Map 1")

    def view2(self):
        type5 = self.ui2.Map2Label.text()

        if type5 == 'Quasi 2D':
            print(type5)
            modelPATH = r'./quasi2d.h5'
            self.view_qmap(dataPATH, modelPATH, 2)
        if type5 == '2D Map':
            print(type5)
            modelPATH = r'./2d.h5'
            self.view_2D(dataPATH, modelPATH)
        if type5 == '3D Map':
            print(type5)
        if type5 == 'Select Map Type':
            self.ui2.Map1Label.setText("Map 1")


#Initialize the App
app = QApplication(sys.argv)
UIWindow = UI()
app.exec_()
