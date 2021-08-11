# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_mainWindow(object):
    def setupUi(self, mainWindow):
        mainWindow.setObjectName("mainWindow")
        mainWindow.resize(1147, 837)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        mainWindow.setFont(font)
        self.centralwidget = QtWidgets.QWidget(mainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.mdiArea = QtWidgets.QMdiArea(self.centralwidget)
        self.mdiArea.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.mdiArea.setObjectName("mdiArea")
        self.horizontalLayout.addWidget(self.mdiArea)
        mainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(mainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1147, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuWindows = QtWidgets.QMenu(self.menubar)
        self.menuWindows.setObjectName("menuWindows")
        self.menuOptions = QtWidgets.QMenu(self.menubar)
        self.menuOptions.setObjectName("menuOptions")
        mainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(mainWindow)
        self.statusbar.setObjectName("statusbar")
        mainWindow.setStatusBar(self.statusbar)
        self.dockWidget_explorer = QtWidgets.QDockWidget(mainWindow)
        self.dockWidget_explorer.setMaximumSize(QtCore.QSize(524287, 524287))
        self.dockWidget_explorer.setObjectName("dockWidget_explorer")
        self.dockWidgetContents = QtWidgets.QWidget()
        self.dockWidgetContents.setObjectName("dockWidgetContents")
        self.gridLayout = QtWidgets.QGridLayout(self.dockWidgetContents)
        self.gridLayout.setObjectName("gridLayout")
        self.tableView_stlModels = QtWidgets.QTableView(self.dockWidgetContents)
        self.tableView_stlModels.setObjectName("tableView_stlModels")
        self.gridLayout.addWidget(self.tableView_stlModels, 4, 0, 1, 2)
        self.pushButton_importModels = QtWidgets.QPushButton(self.dockWidgetContents)
        self.pushButton_importModels.setObjectName("pushButton_importModels")
        self.gridLayout.addWidget(self.pushButton_importModels, 2, 0, 1, 1)
        self.pushButton_modifyModel = QtWidgets.QPushButton(self.dockWidgetContents)
        self.pushButton_modifyModel.setObjectName("pushButton_modifyModel")
        self.gridLayout.addWidget(self.pushButton_modifyModel, 7, 1, 1, 1)
        self.tableView_mathModels = QtWidgets.QTableView(self.dockWidgetContents)
        self.tableView_mathModels.setObjectName("tableView_mathModels")
        self.gridLayout.addWidget(self.tableView_mathModels, 5, 0, 1, 2)
        self.pushButton_deleteModels = QtWidgets.QPushButton(self.dockWidgetContents)
        self.pushButton_deleteModels.setObjectName("pushButton_deleteModels")
        self.gridLayout.addWidget(self.pushButton_deleteModels, 2, 1, 1, 1)
        self.label_projectName = QtWidgets.QLabel(self.dockWidgetContents)
        self.label_projectName.setObjectName("label_projectName")
        self.gridLayout.addWidget(self.label_projectName, 0, 0, 1, 2)
        self.pushButton_showAllModels = QtWidgets.QPushButton(self.dockWidgetContents)
        self.pushButton_showAllModels.setObjectName("pushButton_showAllModels")
        self.gridLayout.addWidget(self.pushButton_showAllModels, 3, 0, 1, 1)
        self.pushButton_showSelectedModels = QtWidgets.QPushButton(self.dockWidgetContents)
        self.pushButton_showSelectedModels.setObjectName("pushButton_showSelectedModels")
        self.gridLayout.addWidget(self.pushButton_showSelectedModels, 3, 1, 1, 1)
        self.dockWidget_explorer.setWidget(self.dockWidgetContents)
        mainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(1), self.dockWidget_explorer)
        self.dockWidget_console = QtWidgets.QDockWidget(mainWindow)
        self.dockWidget_console.setObjectName("dockWidget_console")
        self.dockWidgetContents_2 = QtWidgets.QWidget()
        self.dockWidgetContents_2.setObjectName("dockWidgetContents_2")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.dockWidgetContents_2)
        self.verticalLayout.setObjectName("verticalLayout")
        self.textEdit_console = QtWidgets.QTextEdit(self.dockWidgetContents_2)
        self.textEdit_console.setReadOnly(True)
        self.textEdit_console.setObjectName("textEdit_console")
        self.verticalLayout.addWidget(self.textEdit_console)
        self.dockWidget_console.setWidget(self.dockWidgetContents_2)
        mainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(8), self.dockWidget_console)
        self.dockWidget_controlPanel = QtWidgets.QDockWidget(mainWindow)
        self.dockWidget_controlPanel.setMinimumSize(QtCore.QSize(300, 377))
        self.dockWidget_controlPanel.setMaximumSize(QtCore.QSize(300, 400))
        self.dockWidget_controlPanel.setObjectName("dockWidget_controlPanel")
        self.dockWidgetContents_3 = QtWidgets.QWidget()
        self.dockWidgetContents_3.setObjectName("dockWidgetContents_3")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.dockWidgetContents_3)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.groupBox_genPoints = QtWidgets.QGroupBox(self.dockWidgetContents_3)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.groupBox_genPoints.setFont(font)
        self.groupBox_genPoints.setObjectName("groupBox_genPoints")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox_genPoints)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.lineEdit_interval = QtWidgets.QLineEdit(self.groupBox_genPoints)
        self.lineEdit_interval.setObjectName("lineEdit_interval")
        self.gridLayout_2.addWidget(self.lineEdit_interval, 3, 1, 1, 2)
        self.pushButton_genLatticeModel = QtWidgets.QPushButton(self.groupBox_genPoints)
        self.pushButton_genLatticeModel.setObjectName("pushButton_genLatticeModel")
        self.gridLayout_2.addWidget(self.pushButton_genLatticeModel, 4, 2, 1, 1)
        self.lineEdit_gridPointsNum = QtWidgets.QLineEdit(self.groupBox_genPoints)
        self.lineEdit_gridPointsNum.setObjectName("lineEdit_gridPointsNum")
        self.gridLayout_2.addWidget(self.lineEdit_gridPointsNum, 2, 2, 1, 1)
        self.radioButton_gridPointsNum = QtWidgets.QRadioButton(self.groupBox_genPoints)
        self.radioButton_gridPointsNum.setObjectName("radioButton_gridPointsNum")
        self.gridLayout_2.addWidget(self.radioButton_gridPointsNum, 2, 0, 1, 1)
        self.radioButton_interval = QtWidgets.QRadioButton(self.groupBox_genPoints)
        self.radioButton_interval.setObjectName("radioButton_interval")
        self.gridLayout_2.addWidget(self.radioButton_interval, 3, 0, 1, 1)
        self.verticalLayout_2.addWidget(self.groupBox_genPoints)
        self.groupBox_calcSas = QtWidgets.QGroupBox(self.dockWidgetContents_3)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.groupBox_calcSas.setFont(font)
        self.groupBox_calcSas.setObjectName("groupBox_calcSas")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.groupBox_calcSas)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label_4 = QtWidgets.QLabel(self.groupBox_calcSas)
        self.label_4.setObjectName("label_4")
        self.gridLayout_3.addWidget(self.label_4, 1, 0, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.groupBox_calcSas)
        self.label_3.setObjectName("label_3")
        self.gridLayout_3.addWidget(self.label_3, 0, 0, 1, 1)
        self.lineEdit_qmin = QtWidgets.QLineEdit(self.groupBox_calcSas)
        self.lineEdit_qmin.setObjectName("lineEdit_qmin")
        self.gridLayout_3.addWidget(self.lineEdit_qmin, 0, 2, 1, 1)
        self.lineEdit_lmax = QtWidgets.QLineEdit(self.groupBox_calcSas)
        self.lineEdit_lmax.setObjectName("lineEdit_lmax")
        self.gridLayout_3.addWidget(self.lineEdit_lmax, 2, 2, 1, 2)
        self.lineEdit_qmax = QtWidgets.QLineEdit(self.groupBox_calcSas)
        self.lineEdit_qmax.setObjectName("lineEdit_qmax")
        self.gridLayout_3.addWidget(self.lineEdit_qmax, 0, 3, 1, 1)
        self.pushButton_calcSas = QtWidgets.QPushButton(self.groupBox_calcSas)
        self.pushButton_calcSas.setObjectName("pushButton_calcSas")
        self.gridLayout_3.addWidget(self.pushButton_calcSas, 8, 3, 1, 1)
        self.lineEdit_qnum = QtWidgets.QLineEdit(self.groupBox_calcSas)
        self.lineEdit_qnum.setObjectName("lineEdit_qnum")
        self.gridLayout_3.addWidget(self.lineEdit_qnum, 1, 2, 1, 2)
        self.label_5 = QtWidgets.QLabel(self.groupBox_calcSas)
        self.label_5.setObjectName("label_5")
        self.gridLayout_3.addWidget(self.label_5, 2, 0, 1, 1)
        self.checkBox_useGpu = QtWidgets.QCheckBox(self.groupBox_calcSas)
        self.checkBox_useGpu.setObjectName("checkBox_useGpu")
        self.gridLayout_3.addWidget(self.checkBox_useGpu, 8, 2, 1, 1)
        self.verticalLayout_2.addWidget(self.groupBox_calcSas)
        self.progressBar = QtWidgets.QProgressBar(self.dockWidgetContents_3)
        self.progressBar.setMaximum(100)
        self.progressBar.setProperty("value", -1)
        self.progressBar.setObjectName("progressBar")
        self.verticalLayout_2.addWidget(self.progressBar)
        self.dockWidget_controlPanel.setWidget(self.dockWidgetContents_3)
        mainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(1), self.dockWidget_controlPanel)
        self.actionLoad = QtWidgets.QAction(mainWindow)
        self.actionLoad.setObjectName("actionLoad")
        self.action_saveProject = QtWidgets.QAction(mainWindow)
        self.action_saveProject.setObjectName("action_saveProject")
        self.action_importModels = QtWidgets.QAction(mainWindow)
        self.action_importModels.setObjectName("action_importModels")
        self.actionCascade = QtWidgets.QAction(mainWindow)
        self.actionCascade.setObjectName("actionCascade")
        self.actionTile = QtWidgets.QAction(mainWindow)
        self.actionTile.setObjectName("actionTile")
        self.action_Cascade = QtWidgets.QAction(mainWindow)
        self.action_Cascade.setObjectName("action_Cascade")
        self.action_Tile = QtWidgets.QAction(mainWindow)
        self.action_Tile.setObjectName("action_Tile")
        self.actionControl_Panel = QtWidgets.QAction(mainWindow)
        self.actionControl_Panel.setObjectName("actionControl_Panel")
        self.action_newProject = QtWidgets.QAction(mainWindow)
        self.action_newProject.setObjectName("action_newProject")
        self.action_deleteAllModels = QtWidgets.QAction(mainWindow)
        self.action_deleteAllModels.setObjectName("action_deleteAllModels")
        self.action_saveLatticeModel = QtWidgets.QAction(mainWindow)
        self.action_saveLatticeModel.setObjectName("action_saveLatticeModel")
        self.action_saveSasCurve = QtWidgets.QAction(mainWindow)
        self.action_saveSasCurve.setObjectName("action_saveSasCurve")
        self.action_loadProject = QtWidgets.QAction(mainWindow)
        self.action_loadProject.setObjectName("action_loadProject")
        self.action_configureGpu = QtWidgets.QAction(mainWindow)
        self.action_configureGpu.setObjectName("action_configureGpu")
        self.action_showLatticeModel = QtWidgets.QAction(mainWindow)
        self.action_showLatticeModel.setObjectName("action_showLatticeModel")
        self.action_showSasCurve = QtWidgets.QAction(mainWindow)
        self.action_showSasCurve.setObjectName("action_showSasCurve")
        self.dockWidget_controlPanel.raise_()
        self.menuFile.addAction(self.action_newProject)
        self.menuFile.addAction(self.action_loadProject)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.action_saveProject)
        self.menuFile.addAction(self.action_saveLatticeModel)
        self.menuFile.addAction(self.action_saveSasCurve)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.action_importModels)
        self.menuFile.addAction(self.action_deleteAllModels)
        self.menuWindows.addAction(self.action_showLatticeModel)
        self.menuWindows.addAction(self.action_showSasCurve)
        self.menuWindows.addSeparator()
        self.menuWindows.addAction(self.action_Cascade)
        self.menuWindows.addAction(self.action_Tile)
        self.menuOptions.addAction(self.action_configureGpu)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuOptions.menuAction())
        self.menubar.addAction(self.menuWindows.menuAction())

        self.retranslateUi(mainWindow)
        QtCore.QMetaObject.connectSlotsByName(mainWindow)

    def retranslateUi(self, mainWindow):
        _translate = QtCore.QCoreApplication.translate
        mainWindow.setWindowTitle(_translate("mainWindow", "Model2SAS"))
        self.menuFile.setTitle(_translate("mainWindow", "File"))
        self.menuWindows.setTitle(_translate("mainWindow", "Windows"))
        self.menuOptions.setTitle(_translate("mainWindow", "Options"))
        self.dockWidget_explorer.setWindowTitle(_translate("mainWindow", "Model Explorer"))
        self.pushButton_importModels.setText(_translate("mainWindow", "import model(s)"))
        self.pushButton_modifyModel.setText(_translate("mainWindow", "modify model"))
        self.pushButton_deleteModels.setText(_translate("mainWindow", "delete selected model(s)"))
        self.label_projectName.setText(_translate("mainWindow", "Project: New Project"))
        self.pushButton_showAllModels.setText(_translate("mainWindow", "show all models"))
        self.pushButton_showSelectedModels.setText(_translate("mainWindow", "show selected model(s)"))
        self.dockWidget_console.setWindowTitle(_translate("mainWindow", "Console"))
        self.dockWidget_controlPanel.setWindowTitle(_translate("mainWindow", "Control Panel"))
        self.groupBox_genPoints.setTitle(_translate("mainWindow", "Generate lattice model"))
        self.lineEdit_interval.setText(_translate("mainWindow", "1.0"))
        self.pushButton_genLatticeModel.setText(_translate("mainWindow", "generate"))
        self.lineEdit_gridPointsNum.setText(_translate("mainWindow", "10000"))
        self.radioButton_gridPointsNum.setText(_translate("mainWindow", "grid points number"))
        self.radioButton_interval.setText(_translate("mainWindow", "interval"))
        self.groupBox_calcSas.setTitle(_translate("mainWindow", "Calculate SAS data"))
        self.label_4.setText(_translate("mainWindow", "q number"))
        self.label_3.setText(_translate("mainWindow", "q_min, q_max"))
        self.lineEdit_qmin.setText(_translate("mainWindow", "0.001"))
        self.lineEdit_lmax.setText(_translate("mainWindow", "50"))
        self.lineEdit_qmax.setText(_translate("mainWindow", "1"))
        self.pushButton_calcSas.setText(_translate("mainWindow", "calculate"))
        self.lineEdit_qnum.setText(_translate("mainWindow", "200"))
        self.label_5.setText(_translate("mainWindow", "l_max"))
        self.checkBox_useGpu.setText(_translate("mainWindow", "use GPU"))
        self.actionLoad.setText(_translate("mainWindow", "Load Project"))
        self.action_saveProject.setText(_translate("mainWindow", "Save project"))
        self.action_importModels.setText(_translate("mainWindow", "Import model(s)"))
        self.actionCascade.setText(_translate("mainWindow", "Cascade"))
        self.actionTile.setText(_translate("mainWindow", "Tile"))
        self.action_Cascade.setText(_translate("mainWindow", "Cascade"))
        self.action_Tile.setText(_translate("mainWindow", "Tile"))
        self.actionControl_Panel.setText(_translate("mainWindow", "Contol Panel"))
        self.action_newProject.setText(_translate("mainWindow", "New project"))
        self.action_deleteAllModels.setText(_translate("mainWindow", "Delete all models"))
        self.action_saveLatticeModel.setText(_translate("mainWindow", "Save lattice model"))
        self.action_saveSasCurve.setText(_translate("mainWindow", "Save SAS curve"))
        self.action_loadProject.setText(_translate("mainWindow", "Load project"))
        self.action_configureGpu.setText(_translate("mainWindow", "Configure GPU"))
        self.action_showLatticeModel.setText(_translate("mainWindow", "Show lattice model"))
        self.action_showSasCurve.setText(_translate("mainWindow", "Show SAS curve"))
