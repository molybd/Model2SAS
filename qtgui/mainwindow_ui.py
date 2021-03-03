# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_mainWindow(object):
    def setupUi(self, mainWindow):
        mainWindow.setObjectName("mainWindow")
        mainWindow.resize(1224, 914)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        mainWindow.setFont(font)
        self.centralwidget = QtWidgets.QWidget(mainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.mdiArea = QtWidgets.QMdiArea(self.centralwidget)
        self.mdiArea.setObjectName("mdiArea")
        self.horizontalLayout.addWidget(self.mdiArea)
        mainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(mainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1224, 23))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuWindows = QtWidgets.QMenu(self.menubar)
        self.menuWindows.setObjectName("menuWindows")
        mainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(mainWindow)
        self.statusbar.setObjectName("statusbar")
        mainWindow.setStatusBar(self.statusbar)
        self.dockWidget_explorer = QtWidgets.QDockWidget(mainWindow)
        self.dockWidget_explorer.setObjectName("dockWidget_explorer")
        self.dockWidgetContents = QtWidgets.QWidget()
        self.dockWidgetContents.setObjectName("dockWidgetContents")
        self.gridLayout = QtWidgets.QGridLayout(self.dockWidgetContents)
        self.gridLayout.setObjectName("gridLayout")
        self.tableView_stlmodels = QtWidgets.QTableView(self.dockWidgetContents)
        self.tableView_stlmodels.setObjectName("tableView_stlmodels")
        self.gridLayout.addWidget(self.tableView_stlmodels, 1, 0, 1, 2)
        self.tableView_mathmodels = QtWidgets.QTableView(self.dockWidgetContents)
        self.tableView_mathmodels.setObjectName("tableView_mathmodels")
        self.gridLayout.addWidget(self.tableView_mathmodels, 1, 2, 1, 2)
        self.pushButton_showMathmodel = QtWidgets.QPushButton(self.dockWidgetContents)
        self.pushButton_showMathmodel.setObjectName("pushButton_showMathmodel")
        self.gridLayout.addWidget(self.pushButton_showMathmodel, 2, 3, 1, 1)
        self.pushButton_showStlmodels = QtWidgets.QPushButton(self.dockWidgetContents)
        self.pushButton_showStlmodels.setObjectName("pushButton_showStlmodels")
        self.gridLayout.addWidget(self.pushButton_showStlmodels, 2, 1, 1, 1)
        self.pushButton_deleteModel = QtWidgets.QPushButton(self.dockWidgetContents)
        self.pushButton_deleteModel.setObjectName("pushButton_deleteModel")
        self.gridLayout.addWidget(self.pushButton_deleteModel, 0, 3, 1, 1)
        self.pushButton_importModels = QtWidgets.QPushButton(self.dockWidgetContents)
        self.pushButton_importModels.setObjectName("pushButton_importModels")
        self.gridLayout.addWidget(self.pushButton_importModels, 0, 2, 1, 1)
        self.label_projectName = QtWidgets.QLabel(self.dockWidgetContents)
        self.label_projectName.setObjectName("label_projectName")
        self.gridLayout.addWidget(self.label_projectName, 0, 0, 1, 2)
        self.dockWidget_explorer.setWidget(self.dockWidgetContents)
        mainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(1), self.dockWidget_explorer)
        self.dockWidget_console = QtWidgets.QDockWidget(mainWindow)
        self.dockWidget_console.setObjectName("dockWidget_console")
        self.dockWidgetContents_2 = QtWidgets.QWidget()
        self.dockWidgetContents_2.setObjectName("dockWidgetContents_2")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.dockWidgetContents_2)
        self.verticalLayout.setObjectName("verticalLayout")
        self.textEdit = QtWidgets.QTextEdit(self.dockWidgetContents_2)
        self.textEdit.setReadOnly(True)
        self.textEdit.setObjectName("textEdit")
        self.verticalLayout.addWidget(self.textEdit)
        self.progressBar = QtWidgets.QProgressBar(self.dockWidgetContents_2)
        self.progressBar.setMaximum(100)
        self.progressBar.setProperty("value", -1)
        self.progressBar.setObjectName("progressBar")
        self.verticalLayout.addWidget(self.progressBar)
        self.dockWidget_console.setWidget(self.dockWidgetContents_2)
        mainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(8), self.dockWidget_console)
        self.dockWidget_controlPanel = QtWidgets.QDockWidget(mainWindow)
        self.dockWidget_controlPanel.setObjectName("dockWidget_controlPanel")
        self.dockWidgetContents_3 = QtWidgets.QWidget()
        self.dockWidgetContents_3.setObjectName("dockWidgetContents_3")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.dockWidgetContents_3)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.groupBox_genPoints = QtWidgets.QGroupBox(self.dockWidgetContents_3)
        self.groupBox_genPoints.setObjectName("groupBox_genPoints")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox_genPoints)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.pushButton_genPoints = QtWidgets.QPushButton(self.groupBox_genPoints)
        self.pushButton_genPoints.setObjectName("pushButton_genPoints")
        self.gridLayout_2.addWidget(self.pushButton_genPoints, 2, 2, 1, 1)
        self.lineEdit_interval = QtWidgets.QLineEdit(self.groupBox_genPoints)
        self.lineEdit_interval.setObjectName("lineEdit_interval")
        self.gridLayout_2.addWidget(self.lineEdit_interval, 1, 1, 1, 2)
        self.lineEdit_gridPointsNum = QtWidgets.QLineEdit(self.groupBox_genPoints)
        self.lineEdit_gridPointsNum.setObjectName("lineEdit_gridPointsNum")
        self.gridLayout_2.addWidget(self.lineEdit_gridPointsNum, 0, 2, 1, 1)
        self.label = QtWidgets.QLabel(self.groupBox_genPoints)
        self.label.setObjectName("label")
        self.gridLayout_2.addWidget(self.label, 0, 0, 1, 2)
        self.label_2 = QtWidgets.QLabel(self.groupBox_genPoints)
        self.label_2.setObjectName("label_2")
        self.gridLayout_2.addWidget(self.label_2, 1, 0, 1, 1)
        self.verticalLayout_2.addWidget(self.groupBox_genPoints)
        self.groupBox_calcSas = QtWidgets.QGroupBox(self.dockWidgetContents_3)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.groupBox_calcSas.setFont(font)
        self.groupBox_calcSas.setObjectName("groupBox_calcSas")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.groupBox_calcSas)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.lineEdit_lmax = QtWidgets.QLineEdit(self.groupBox_calcSas)
        self.lineEdit_lmax.setObjectName("lineEdit_lmax")
        self.gridLayout_3.addWidget(self.lineEdit_lmax, 2, 1, 1, 2)
        self.label_6 = QtWidgets.QLabel(self.groupBox_calcSas)
        self.label_6.setObjectName("label_6")
        self.gridLayout_3.addWidget(self.label_6, 6, 0, 1, 1)
        self.lineEdit_qnum = QtWidgets.QLineEdit(self.groupBox_calcSas)
        self.lineEdit_qnum.setObjectName("lineEdit_qnum")
        self.gridLayout_3.addWidget(self.lineEdit_qnum, 1, 1, 1, 2)
        self.label_4 = QtWidgets.QLabel(self.groupBox_calcSas)
        self.label_4.setObjectName("label_4")
        self.gridLayout_3.addWidget(self.label_4, 1, 0, 1, 1)
        self.lineEdit_qmin = QtWidgets.QLineEdit(self.groupBox_calcSas)
        self.lineEdit_qmin.setObjectName("lineEdit_qmin")
        self.gridLayout_3.addWidget(self.lineEdit_qmin, 0, 1, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.groupBox_calcSas)
        self.label_3.setObjectName("label_3")
        self.gridLayout_3.addWidget(self.label_3, 0, 0, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.groupBox_calcSas)
        self.label_5.setObjectName("label_5")
        self.gridLayout_3.addWidget(self.label_5, 2, 0, 1, 1)
        self.lineEdit_qmax = QtWidgets.QLineEdit(self.groupBox_calcSas)
        self.lineEdit_qmax.setObjectName("lineEdit_qmax")
        self.gridLayout_3.addWidget(self.lineEdit_qmax, 0, 2, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.groupBox_calcSas)
        self.label_7.setObjectName("label_7")
        self.gridLayout_3.addWidget(self.label_7, 7, 0, 1, 1)
        self.checkBox_parallel = QtWidgets.QCheckBox(self.groupBox_calcSas)
        self.checkBox_parallel.setEnabled(True)
        self.checkBox_parallel.setChecked(False)
        self.checkBox_parallel.setObjectName("checkBox_parallel")
        self.gridLayout_3.addWidget(self.checkBox_parallel, 4, 0, 1, 1)
        self.pushButton_calcSas = QtWidgets.QPushButton(self.groupBox_calcSas)
        self.pushButton_calcSas.setObjectName("pushButton_calcSas")
        self.gridLayout_3.addWidget(self.pushButton_calcSas, 4, 1, 1, 2)
        self.lineEdit_cpuUsage = QtWidgets.QLineEdit(self.groupBox_calcSas)
        self.lineEdit_cpuUsage.setObjectName("lineEdit_cpuUsage")
        self.gridLayout_3.addWidget(self.lineEdit_cpuUsage, 6, 1, 1, 1)
        self.lineEdit_processNum = QtWidgets.QLineEdit(self.groupBox_calcSas)
        self.lineEdit_processNum.setObjectName("lineEdit_processNum")
        self.gridLayout_3.addWidget(self.lineEdit_processNum, 7, 1, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.groupBox_calcSas)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.gridLayout_3.addWidget(self.label_8, 5, 0, 1, 1)
        self.verticalLayout_2.addWidget(self.groupBox_calcSas)
        self.dockWidget_controlPanel.setWidget(self.dockWidgetContents_3)
        mainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(1), self.dockWidget_controlPanel)
        self.actionLoad = QtWidgets.QAction(mainWindow)
        self.actionLoad.setObjectName("actionLoad")
        self.actionSave_project = QtWidgets.QAction(mainWindow)
        self.actionSave_project.setObjectName("actionSave_project")
        self.actionImport_model_s = QtWidgets.QAction(mainWindow)
        self.actionImport_model_s.setObjectName("actionImport_model_s")
        self.actionCascade = QtWidgets.QAction(mainWindow)
        self.actionCascade.setObjectName("actionCascade")
        self.actionTile = QtWidgets.QAction(mainWindow)
        self.actionTile.setObjectName("actionTile")
        self.actionCascade_2 = QtWidgets.QAction(mainWindow)
        self.actionCascade_2.setObjectName("actionCascade_2")
        self.actionTile_2 = QtWidgets.QAction(mainWindow)
        self.actionTile_2.setObjectName("actionTile_2")
        self.actionControl_Panel = QtWidgets.QAction(mainWindow)
        self.actionControl_Panel.setObjectName("actionControl_Panel")
        self.actionNew_Project = QtWidgets.QAction(mainWindow)
        self.actionNew_Project.setObjectName("actionNew_Project")
        self.actionDelete_all_models = QtWidgets.QAction(mainWindow)
        self.actionDelete_all_models.setObjectName("actionDelete_all_models")
        self.actionSave_points_model = QtWidgets.QAction(mainWindow)
        self.actionSave_points_model.setObjectName("actionSave_points_model")
        self.actionSave_SAS_curve = QtWidgets.QAction(mainWindow)
        self.actionSave_SAS_curve.setObjectName("actionSave_SAS_curve")
        self.dockWidget_controlPanel.raise_()
        self.menuFile.addAction(self.actionNew_Project)
        self.menuFile.addAction(self.actionSave_project)
        self.menuFile.addAction(self.actionSave_points_model)
        self.menuFile.addAction(self.actionSave_SAS_curve)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionImport_model_s)
        self.menuFile.addAction(self.actionDelete_all_models)
        self.menuWindows.addAction(self.actionCascade_2)
        self.menuWindows.addAction(self.actionTile_2)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuWindows.menuAction())

        self.retranslateUi(mainWindow)
        QtCore.QMetaObject.connectSlotsByName(mainWindow)

    def retranslateUi(self, mainWindow):
        _translate = QtCore.QCoreApplication.translate
        mainWindow.setWindowTitle(_translate("mainWindow", "Model2SAS"))
        self.menuFile.setTitle(_translate("mainWindow", "File"))
        self.menuWindows.setTitle(_translate("mainWindow", "Windows"))
        self.dockWidget_explorer.setWindowTitle(_translate("mainWindow", "Model Explorer"))
        self.pushButton_showMathmodel.setText(_translate("mainWindow", "show math model"))
        self.pushButton_showStlmodels.setText(_translate("mainWindow", "show stl model(s)"))
        self.pushButton_deleteModel.setText(_translate("mainWindow", "delete selected model"))
        self.pushButton_importModels.setText(_translate("mainWindow", "import model(s)"))
        self.label_projectName.setText(_translate("mainWindow", "Project: New Project"))
        self.dockWidget_console.setWindowTitle(_translate("mainWindow", "Console"))
        self.dockWidget_controlPanel.setWindowTitle(_translate("mainWindow", "Control Panel"))
        self.groupBox_genPoints.setTitle(_translate("mainWindow", "generate points model"))
        self.pushButton_genPoints.setText(_translate("mainWindow", "generate"))
        self.lineEdit_gridPointsNum.setText(_translate("mainWindow", "10000"))
        self.label.setText(_translate("mainWindow", "grid points number"))
        self.label_2.setText(_translate("mainWindow", "OR interval"))
        self.groupBox_calcSas.setTitle(_translate("mainWindow", "calculate SAS data"))
        self.lineEdit_lmax.setText(_translate("mainWindow", "50"))
        self.label_6.setText(_translate("mainWindow", "CPU usage"))
        self.lineEdit_qnum.setText(_translate("mainWindow", "200"))
        self.label_4.setText(_translate("mainWindow", "q number"))
        self.lineEdit_qmin.setText(_translate("mainWindow", "0.001"))
        self.label_3.setText(_translate("mainWindow", "q_min, q_max"))
        self.label_5.setText(_translate("mainWindow", "l_max"))
        self.lineEdit_qmax.setText(_translate("mainWindow", "1"))
        self.label_7.setText(_translate("mainWindow", "OR process num"))
        self.checkBox_parallel.setText(_translate("mainWindow", "parallel"))
        self.pushButton_calcSas.setText(_translate("mainWindow", "calculate"))
        self.lineEdit_cpuUsage.setText(_translate("mainWindow", "0.4"))
        self.label_8.setText(_translate("mainWindow", "Parallel Control"))
        self.actionLoad.setText(_translate("mainWindow", "Load Project"))
        self.actionSave_project.setText(_translate("mainWindow", "Save project"))
        self.actionImport_model_s.setText(_translate("mainWindow", "Import model(s)"))
        self.actionCascade.setText(_translate("mainWindow", "Cascade"))
        self.actionTile.setText(_translate("mainWindow", "Tile"))
        self.actionCascade_2.setText(_translate("mainWindow", "Cascade"))
        self.actionTile_2.setText(_translate("mainWindow", "Tile"))
        self.actionControl_Panel.setText(_translate("mainWindow", "Contol Panel"))
        self.actionNew_Project.setText(_translate("mainWindow", "New project"))
        self.actionDelete_all_models.setText(_translate("mainWindow", "Delete all models"))
        self.actionSave_points_model.setText(_translate("mainWindow", "Save points model"))
        self.actionSave_SAS_curve.setText(_translate("mainWindow", "Save SAS curve"))
