# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'MainWindow.ui'
##
## Created by: Qt User Interface Compiler version 6.5.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDockWidget,
    QGridLayout, QGroupBox, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QMainWindow, QMdiArea,
    QMenuBar, QPushButton, QSizePolicy, QSpacerItem,
    QStatusBar, QTabWidget, QTableView, QTextBrowser,
    QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1068, 810)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.mdiArea = QMdiArea(self.centralwidget)
        self.mdiArea.setObjectName(u"mdiArea")

        self.gridLayout.addWidget(self.mdiArea, 0, 0, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1068, 22))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.dockWidget_main = QDockWidget(MainWindow)
        self.dockWidget_main.setObjectName(u"dockWidget_main")
        self.dockWidget_main.setFeatures(QDockWidget.DockWidgetClosable|QDockWidget.DockWidgetFloatable|QDockWidget.DockWidgetMovable)
        self.dockWidgetContents = QWidget()
        self.dockWidgetContents.setObjectName(u"dockWidgetContents")
        self.gridLayout_7 = QGridLayout(self.dockWidgetContents)
        self.gridLayout_7.setObjectName(u"gridLayout_7")
        self.treeWidget_models = QTreeWidget(self.dockWidgetContents)
        __qtreewidgetitem = QTreeWidgetItem()
        __qtreewidgetitem.setText(0, u"1");
        self.treeWidget_models.setHeaderItem(__qtreewidgetitem)
        self.treeWidget_models.setObjectName(u"treeWidget_models")

        self.gridLayout_7.addWidget(self.treeWidget_models, 2, 0, 1, 1)

        self.groupBox_part = QGroupBox(self.dockWidgetContents)
        self.groupBox_part.setObjectName(u"groupBox_part")
        self.gridLayout_5 = QGridLayout(self.groupBox_part)
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.label_part_device = QLabel(self.groupBox_part)
        self.label_part_device.setObjectName(u"label_part_device")

        self.gridLayout_5.addWidget(self.label_part_device, 0, 0, 1, 1)

        self.comboBox_part_device = QComboBox(self.groupBox_part)
        self.comboBox_part_device.addItem("")
        self.comboBox_part_device.setObjectName(u"comboBox_part_device")

        self.gridLayout_5.addWidget(self.comboBox_part_device, 0, 1, 1, 1)

        self.pushButton_part_from_files = QPushButton(self.groupBox_part)
        self.pushButton_part_from_files.setObjectName(u"pushButton_part_from_files")

        self.gridLayout_5.addWidget(self.pushButton_part_from_files, 1, 0, 1, 2)

        self.pushButton_build_math_model = QPushButton(self.groupBox_part)
        self.pushButton_build_math_model.setObjectName(u"pushButton_build_math_model")

        self.gridLayout_5.addWidget(self.pushButton_build_math_model, 2, 0, 1, 2)


        self.gridLayout_7.addWidget(self.groupBox_part, 0, 0, 1, 1)

        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.label_length_unit = QLabel(self.dockWidgetContents)
        self.label_length_unit.setObjectName(u"label_length_unit")

        self.horizontalLayout_7.addWidget(self.label_length_unit)

        self.comboBox_length_unit = QComboBox(self.dockWidgetContents)
        self.comboBox_length_unit.addItem("")
        self.comboBox_length_unit.setObjectName(u"comboBox_length_unit")

        self.horizontalLayout_7.addWidget(self.comboBox_length_unit)


        self.gridLayout_7.addLayout(self.horizontalLayout_7, 3, 0, 1, 1)

        self.groupBox_assembly = QGroupBox(self.dockWidgetContents)
        self.groupBox_assembly.setObjectName(u"groupBox_assembly")
        self.gridLayout_6 = QGridLayout(self.groupBox_assembly)
        self.gridLayout_6.setObjectName(u"gridLayout_6")
        self.label_assembly_device = QLabel(self.groupBox_assembly)
        self.label_assembly_device.setObjectName(u"label_assembly_device")

        self.gridLayout_6.addWidget(self.label_assembly_device, 0, 0, 1, 1)

        self.comboBox_assembly_device = QComboBox(self.groupBox_assembly)
        self.comboBox_assembly_device.addItem("")
        self.comboBox_assembly_device.setObjectName(u"comboBox_assembly_device")

        self.gridLayout_6.addWidget(self.comboBox_assembly_device, 0, 1, 1, 1)

        self.pushButton_add_to_assembly = QPushButton(self.groupBox_assembly)
        self.pushButton_add_to_assembly.setObjectName(u"pushButton_add_to_assembly")

        self.gridLayout_6.addWidget(self.pushButton_add_to_assembly, 2, 0, 1, 2)

        self.pushButton_new_assembly = QPushButton(self.groupBox_assembly)
        self.pushButton_new_assembly.setObjectName(u"pushButton_new_assembly")

        self.gridLayout_6.addWidget(self.pushButton_new_assembly, 1, 0, 1, 2)


        self.gridLayout_7.addWidget(self.groupBox_assembly, 1, 0, 1, 1)

        self.tabWidget_settings = QTabWidget(self.dockWidgetContents)
        self.tabWidget_settings.setObjectName(u"tabWidget_settings")
        self.tabPage_model = QWidget()
        self.tabPage_model.setObjectName(u"tabPage_model")
        self.gridLayout_8 = QGridLayout(self.tabPage_model)
        self.gridLayout_8.setObjectName(u"gridLayout_8")
        self.label_model_params = QLabel(self.tabPage_model)
        self.label_model_params.setObjectName(u"label_model_params")

        self.gridLayout_8.addWidget(self.label_model_params, 0, 0, 1, 2)

        self.tableView_model_params = QTableView(self.tabPage_model)
        self.tableView_model_params.setObjectName(u"tableView_model_params")

        self.gridLayout_8.addWidget(self.tableView_model_params, 1, 0, 1, 4)

        self.label_real_lattice_1d_size = QLabel(self.tabPage_model)
        self.label_real_lattice_1d_size.setObjectName(u"label_real_lattice_1d_size")

        self.gridLayout_8.addWidget(self.label_real_lattice_1d_size, 2, 0, 1, 2)

        self.lineEdit_real_lattice_1d_size = QLineEdit(self.tabPage_model)
        self.lineEdit_real_lattice_1d_size.setObjectName(u"lineEdit_real_lattice_1d_size")

        self.gridLayout_8.addWidget(self.lineEdit_real_lattice_1d_size, 2, 2, 1, 1)

        self.pushButton_sampling = QPushButton(self.tabPage_model)
        self.pushButton_sampling.setObjectName(u"pushButton_sampling")

        self.gridLayout_8.addWidget(self.pushButton_sampling, 2, 3, 1, 1)

        self.groupBox_transform = QGroupBox(self.tabPage_model)
        self.groupBox_transform.setObjectName(u"groupBox_transform")
        self.gridLayout_4 = QGridLayout(self.groupBox_transform)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.comboBox_transform_type = QComboBox(self.groupBox_transform)
        self.comboBox_transform_type.addItem("")
        self.comboBox_transform_type.addItem("")
        self.comboBox_transform_type.setObjectName(u"comboBox_transform_type")

        self.gridLayout_4.addWidget(self.comboBox_transform_type, 0, 0, 1, 2)

        self.pushButton_add_transform = QPushButton(self.groupBox_transform)
        self.pushButton_add_transform.setObjectName(u"pushButton_add_transform")

        self.gridLayout_4.addWidget(self.pushButton_add_transform, 1, 0, 1, 1)

        self.pushButton_delete_transform = QPushButton(self.groupBox_transform)
        self.pushButton_delete_transform.setObjectName(u"pushButton_delete_transform")

        self.gridLayout_4.addWidget(self.pushButton_delete_transform, 1, 1, 1, 1)

        self.tableView_transform = QTableView(self.groupBox_transform)
        self.tableView_transform.setObjectName(u"tableView_transform")

        self.gridLayout_4.addWidget(self.tableView_transform, 2, 0, 1, 2)


        self.gridLayout_8.addWidget(self.groupBox_transform, 3, 0, 1, 4)

        self.checkBox = QCheckBox(self.tabPage_model)
        self.checkBox.setObjectName(u"checkBox")

        self.gridLayout_8.addWidget(self.checkBox, 4, 0, 1, 1)

        self.checkBox_2 = QCheckBox(self.tabPage_model)
        self.checkBox_2.setObjectName(u"checkBox_2")

        self.gridLayout_8.addWidget(self.checkBox_2, 4, 1, 1, 1)

        self.pushButton_show_model = QPushButton(self.tabPage_model)
        self.pushButton_show_model.setObjectName(u"pushButton_show_model")

        self.gridLayout_8.addWidget(self.pushButton_show_model, 4, 2, 1, 2)

        self.tabWidget_settings.addTab(self.tabPage_model, "")
        self.tabPage_scattering = QWidget()
        self.tabPage_scattering.setObjectName(u"tabPage_scattering")
        self.gridLayout_3 = QGridLayout(self.tabPage_scattering)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.label_reciprocal_lattice_1d_size = QLabel(self.tabPage_scattering)
        self.label_reciprocal_lattice_1d_size.setObjectName(u"label_reciprocal_lattice_1d_size")

        self.gridLayout_3.addWidget(self.label_reciprocal_lattice_1d_size, 0, 0, 1, 1)

        self.lineEdit_reciprocal_lattice_1d_size = QLineEdit(self.tabPage_scattering)
        self.lineEdit_reciprocal_lattice_1d_size.setObjectName(u"lineEdit_reciprocal_lattice_1d_size")

        self.gridLayout_3.addWidget(self.lineEdit_reciprocal_lattice_1d_size, 0, 1, 1, 1)

        self.tabWidget_measure = QTabWidget(self.tabPage_scattering)
        self.tabWidget_measure.setObjectName(u"tabWidget_measure")
        self.tabPage_direct_measure = QWidget()
        self.tabPage_direct_measure.setObjectName(u"tabPage_direct_measure")
        self.verticalLayout_3 = QVBoxLayout(self.tabPage_direct_measure)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.groupBox_1d_measure = QGroupBox(self.tabPage_direct_measure)
        self.groupBox_1d_measure.setObjectName(u"groupBox_1d_measure")
        self.groupBox_1d_measure.setCheckable(False)
        self.verticalLayout = QVBoxLayout(self.groupBox_1d_measure)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.label_q1d_range = QLabel(self.groupBox_1d_measure)
        self.label_q1d_range.setObjectName(u"label_q1d_range")

        self.horizontalLayout.addWidget(self.label_q1d_range)

        self.lineEdit_q1d_min = QLineEdit(self.groupBox_1d_measure)
        self.lineEdit_q1d_min.setObjectName(u"lineEdit_q1d_min")

        self.horizontalLayout.addWidget(self.lineEdit_q1d_min)

        self.label_dash = QLabel(self.groupBox_1d_measure)
        self.label_dash.setObjectName(u"label_dash")

        self.horizontalLayout.addWidget(self.label_dash)

        self.lineEdit_q1d_max = QLineEdit(self.groupBox_1d_measure)
        self.lineEdit_q1d_max.setObjectName(u"lineEdit_q1d_max")

        self.horizontalLayout.addWidget(self.lineEdit_q1d_max)

        self.label_q1d_unit = QLabel(self.groupBox_1d_measure)
        self.label_q1d_unit.setObjectName(u"label_q1d_unit")

        self.horizontalLayout.addWidget(self.label_q1d_unit)


        self.verticalLayout.addLayout(self.horizontalLayout)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.label_q1d_num = QLabel(self.groupBox_1d_measure)
        self.label_q1d_num.setObjectName(u"label_q1d_num")

        self.horizontalLayout_2.addWidget(self.label_q1d_num)

        self.lineEdit_q1d_num = QLineEdit(self.groupBox_1d_measure)
        self.lineEdit_q1d_num.setObjectName(u"lineEdit_q1d_num")

        self.horizontalLayout_2.addWidget(self.lineEdit_q1d_num)

        self.checkBox_q1d_log_spaced = QCheckBox(self.groupBox_1d_measure)
        self.checkBox_q1d_log_spaced.setObjectName(u"checkBox_q1d_log_spaced")

        self.horizontalLayout_2.addWidget(self.checkBox_q1d_log_spaced)


        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.pushButton_1d_measure = QPushButton(self.groupBox_1d_measure)
        self.pushButton_1d_measure.setObjectName(u"pushButton_1d_measure")

        self.verticalLayout.addWidget(self.pushButton_1d_measure)


        self.verticalLayout_3.addWidget(self.groupBox_1d_measure)

        self.groupBox_2d_measure = QGroupBox(self.tabPage_direct_measure)
        self.groupBox_2d_measure.setObjectName(u"groupBox_2d_measure")
        self.verticalLayout_2 = QVBoxLayout(self.groupBox_2d_measure)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.label_q2d_resolution = QLabel(self.groupBox_2d_measure)
        self.label_q2d_resolution.setObjectName(u"label_q2d_resolution")

        self.horizontalLayout_3.addWidget(self.label_q2d_resolution)

        self.lineEdit_q2d_resolution_h = QLineEdit(self.groupBox_2d_measure)
        self.lineEdit_q2d_resolution_h.setObjectName(u"lineEdit_q2d_resolution_h")

        self.horizontalLayout_3.addWidget(self.lineEdit_q2d_resolution_h)

        self.label_times = QLabel(self.groupBox_2d_measure)
        self.label_times.setObjectName(u"label_times")

        self.horizontalLayout_3.addWidget(self.label_times)

        self.lineEdit_q2d_resolution_v = QLineEdit(self.groupBox_2d_measure)
        self.lineEdit_q2d_resolution_v.setObjectName(u"lineEdit_q2d_resolution_v")

        self.horizontalLayout_3.addWidget(self.lineEdit_q2d_resolution_v)


        self.verticalLayout_2.addLayout(self.horizontalLayout_3)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.label_q2d_spacing = QLabel(self.groupBox_2d_measure)
        self.label_q2d_spacing.setObjectName(u"label_q2d_spacing")

        self.horizontalLayout_4.addWidget(self.label_q2d_spacing)

        self.lineEdit_q2d_spacing = QLineEdit(self.groupBox_2d_measure)
        self.lineEdit_q2d_spacing.setObjectName(u"lineEdit_q2d_spacing")

        self.horizontalLayout_4.addWidget(self.lineEdit_q2d_spacing)

        self.label_q2d_unit = QLabel(self.groupBox_2d_measure)
        self.label_q2d_unit.setObjectName(u"label_q2d_unit")

        self.horizontalLayout_4.addWidget(self.label_q2d_unit)


        self.verticalLayout_2.addLayout(self.horizontalLayout_4)

        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.label_q2d_normal_vector = QLabel(self.groupBox_2d_measure)
        self.label_q2d_normal_vector.setObjectName(u"label_q2d_normal_vector")

        self.horizontalLayout_6.addWidget(self.label_q2d_normal_vector)

        self.lineEdit_q2d_normal_x = QLineEdit(self.groupBox_2d_measure)
        self.lineEdit_q2d_normal_x.setObjectName(u"lineEdit_q2d_normal_x")
        self.lineEdit_q2d_normal_x.setMaximumSize(QSize(40, 16777215))

        self.horizontalLayout_6.addWidget(self.lineEdit_q2d_normal_x)

        self.lineEdit_q2d_normal_y = QLineEdit(self.groupBox_2d_measure)
        self.lineEdit_q2d_normal_y.setObjectName(u"lineEdit_q2d_normal_y")
        self.lineEdit_q2d_normal_y.setMaximumSize(QSize(40, 16777215))

        self.horizontalLayout_6.addWidget(self.lineEdit_q2d_normal_y)

        self.lineEdit_q2d_normal_z = QLineEdit(self.groupBox_2d_measure)
        self.lineEdit_q2d_normal_z.setObjectName(u"lineEdit_q2d_normal_z")
        self.lineEdit_q2d_normal_z.setMaximumSize(QSize(40, 16777215))

        self.horizontalLayout_6.addWidget(self.lineEdit_q2d_normal_z)


        self.verticalLayout_2.addLayout(self.horizontalLayout_6)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.checkBox_2d_show_3d = QCheckBox(self.groupBox_2d_measure)
        self.checkBox_2d_show_3d.setObjectName(u"checkBox_2d_show_3d")

        self.horizontalLayout_5.addWidget(self.checkBox_2d_show_3d)

        self.pushButton_2d_measure = QPushButton(self.groupBox_2d_measure)
        self.pushButton_2d_measure.setObjectName(u"pushButton_2d_measure")

        self.horizontalLayout_5.addWidget(self.pushButton_2d_measure)


        self.verticalLayout_2.addLayout(self.horizontalLayout_5)


        self.verticalLayout_3.addWidget(self.groupBox_2d_measure)

        self.verticalSpacer = QSpacerItem(228, 132, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_3.addItem(self.verticalSpacer)

        self.tabWidget_measure.addTab(self.tabPage_direct_measure, "")
        self.tabPage_virtual_detector = QWidget()
        self.tabPage_virtual_detector.setObjectName(u"tabPage_virtual_detector")
        self.tabWidget_measure.addTab(self.tabPage_virtual_detector, "")

        self.gridLayout_3.addWidget(self.tabWidget_measure, 2, 0, 1, 2)

        self.pushButton_scattering = QPushButton(self.tabPage_scattering)
        self.pushButton_scattering.setObjectName(u"pushButton_scattering")

        self.gridLayout_3.addWidget(self.pushButton_scattering, 1, 0, 1, 2)

        self.tabWidget_settings.addTab(self.tabPage_scattering, "")

        self.gridLayout_7.addWidget(self.tabWidget_settings, 0, 1, 4, 1)

        self.dockWidget_main.setWidget(self.dockWidgetContents)
        MainWindow.addDockWidget(Qt.LeftDockWidgetArea, self.dockWidget_main)
        self.dockWidget_log = QDockWidget(MainWindow)
        self.dockWidget_log.setObjectName(u"dockWidget_log")
        self.dockWidget_log.setFeatures(QDockWidget.DockWidgetClosable|QDockWidget.DockWidgetFloatable|QDockWidget.DockWidgetMovable)
        self.dockWidgetContents_2 = QWidget()
        self.dockWidgetContents_2.setObjectName(u"dockWidgetContents_2")
        self.gridLayout_2 = QGridLayout(self.dockWidgetContents_2)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.textBrowser_log = QTextBrowser(self.dockWidgetContents_2)
        self.textBrowser_log.setObjectName(u"textBrowser_log")

        self.gridLayout_2.addWidget(self.textBrowser_log, 0, 0, 1, 1)

        self.dockWidget_log.setWidget(self.dockWidgetContents_2)
        MainWindow.addDockWidget(Qt.BottomDockWidgetArea, self.dockWidget_log)
        QWidget.setTabOrder(self.comboBox_part_device, self.pushButton_part_from_files)
        QWidget.setTabOrder(self.pushButton_part_from_files, self.pushButton_build_math_model)
        QWidget.setTabOrder(self.pushButton_build_math_model, self.tabWidget_settings)
        QWidget.setTabOrder(self.tabWidget_settings, self.comboBox_transform_type)
        QWidget.setTabOrder(self.comboBox_transform_type, self.pushButton_add_transform)
        QWidget.setTabOrder(self.pushButton_add_transform, self.tableView_transform)
        QWidget.setTabOrder(self.tableView_transform, self.pushButton_delete_transform)
        QWidget.setTabOrder(self.pushButton_delete_transform, self.lineEdit_real_lattice_1d_size)
        QWidget.setTabOrder(self.lineEdit_real_lattice_1d_size, self.pushButton_show_model)
        QWidget.setTabOrder(self.pushButton_show_model, self.tableView_model_params)
        QWidget.setTabOrder(self.tableView_model_params, self.pushButton_sampling)
        QWidget.setTabOrder(self.pushButton_sampling, self.lineEdit_reciprocal_lattice_1d_size)
        QWidget.setTabOrder(self.lineEdit_reciprocal_lattice_1d_size, self.pushButton_scattering)
        QWidget.setTabOrder(self.pushButton_scattering, self.tabWidget_measure)
        QWidget.setTabOrder(self.tabWidget_measure, self.lineEdit_q1d_min)
        QWidget.setTabOrder(self.lineEdit_q1d_min, self.lineEdit_q1d_max)
        QWidget.setTabOrder(self.lineEdit_q1d_max, self.lineEdit_q1d_num)
        QWidget.setTabOrder(self.lineEdit_q1d_num, self.checkBox_q1d_log_spaced)
        QWidget.setTabOrder(self.checkBox_q1d_log_spaced, self.pushButton_1d_measure)
        QWidget.setTabOrder(self.pushButton_1d_measure, self.lineEdit_q2d_resolution_h)
        QWidget.setTabOrder(self.lineEdit_q2d_resolution_h, self.lineEdit_q2d_resolution_v)
        QWidget.setTabOrder(self.lineEdit_q2d_resolution_v, self.lineEdit_q2d_spacing)
        QWidget.setTabOrder(self.lineEdit_q2d_spacing, self.lineEdit_q2d_normal_x)
        QWidget.setTabOrder(self.lineEdit_q2d_normal_x, self.lineEdit_q2d_normal_y)
        QWidget.setTabOrder(self.lineEdit_q2d_normal_y, self.lineEdit_q2d_normal_z)
        QWidget.setTabOrder(self.lineEdit_q2d_normal_z, self.checkBox_2d_show_3d)
        QWidget.setTabOrder(self.checkBox_2d_show_3d, self.pushButton_2d_measure)
        QWidget.setTabOrder(self.pushButton_2d_measure, self.comboBox_assembly_device)
        QWidget.setTabOrder(self.comboBox_assembly_device, self.pushButton_new_assembly)
        QWidget.setTabOrder(self.pushButton_new_assembly, self.pushButton_add_to_assembly)
        QWidget.setTabOrder(self.pushButton_add_to_assembly, self.treeWidget_models)
        QWidget.setTabOrder(self.treeWidget_models, self.comboBox_length_unit)
        QWidget.setTabOrder(self.comboBox_length_unit, self.textBrowser_log)

        self.retranslateUi(MainWindow)
        self.pushButton_part_from_files.clicked.connect(MainWindow.browse_model_files)
        self.pushButton_build_math_model.clicked.connect(MainWindow.build_math_model)

        self.tabWidget_settings.setCurrentIndex(0)
        self.tabWidget_measure.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Model2SAS", None))
        self.dockWidget_main.setWindowTitle(QCoreApplication.translate("MainWindow", u"Main", None))
        self.groupBox_part.setTitle(QCoreApplication.translate("MainWindow", u"Part Model", None))
        self.label_part_device.setText(QCoreApplication.translate("MainWindow", u"Device", None))
        self.comboBox_part_device.setItemText(0, QCoreApplication.translate("MainWindow", u"cpu", None))

        self.pushButton_part_from_files.setText(QCoreApplication.translate("MainWindow", u"Load From File(s)", None))
        self.pushButton_build_math_model.setText(QCoreApplication.translate("MainWindow", u"Build Math Model", None))
        self.label_length_unit.setText(QCoreApplication.translate("MainWindow", u"Length Unit", None))
        self.comboBox_length_unit.setItemText(0, QCoreApplication.translate("MainWindow", u"\u00c5", None))

        self.groupBox_assembly.setTitle(QCoreApplication.translate("MainWindow", u"Assembly Model", None))
        self.label_assembly_device.setText(QCoreApplication.translate("MainWindow", u"Device", None))
        self.comboBox_assembly_device.setItemText(0, QCoreApplication.translate("MainWindow", u"cpu", None))

        self.pushButton_add_to_assembly.setText(QCoreApplication.translate("MainWindow", u"Add to Assembly Model", None))
        self.pushButton_new_assembly.setText(QCoreApplication.translate("MainWindow", u"New Assembly Model", None))
        self.label_model_params.setText(QCoreApplication.translate("MainWindow", u"Model Parameters", None))
        self.label_real_lattice_1d_size.setText(QCoreApplication.translate("MainWindow", u"Points in longest edge", None))
        self.lineEdit_real_lattice_1d_size.setText(QCoreApplication.translate("MainWindow", u"50", None))
        self.pushButton_sampling.setText(QCoreApplication.translate("MainWindow", u"Sampling", None))
        self.groupBox_transform.setTitle(QCoreApplication.translate("MainWindow", u"Transform", None))
        self.comboBox_transform_type.setItemText(0, QCoreApplication.translate("MainWindow", u"Translate", None))
        self.comboBox_transform_type.setItemText(1, QCoreApplication.translate("MainWindow", u"Rotate", None))

        self.pushButton_add_transform.setText(QCoreApplication.translate("MainWindow", u"Add", None))
        self.pushButton_delete_transform.setText(QCoreApplication.translate("MainWindow", u"Delete", None))
        self.checkBox.setText(QCoreApplication.translate("MainWindow", u"Voxel", None))
        self.checkBox_2.setText(QCoreApplication.translate("MainWindow", u"Volume", None))
        self.pushButton_show_model.setText(QCoreApplication.translate("MainWindow", u"Show", None))
        self.tabWidget_settings.setTabText(self.tabWidget_settings.indexOf(self.tabPage_model), QCoreApplication.translate("MainWindow", u"Model Settings", None))
        self.label_reciprocal_lattice_1d_size.setText(QCoreApplication.translate("MainWindow", u"Edge points in reciprocal grid", None))
        self.lineEdit_reciprocal_lattice_1d_size.setText(QCoreApplication.translate("MainWindow", u"default", None))
        self.groupBox_1d_measure.setTitle(QCoreApplication.translate("MainWindow", u"1D", None))
        self.label_q1d_range.setText(QCoreApplication.translate("MainWindow", u"Q range", None))
        self.label_dash.setText(QCoreApplication.translate("MainWindow", u"-", None))
        self.label_q1d_unit.setText(QCoreApplication.translate("MainWindow", u"\u00c5^-1", None))
        self.label_q1d_num.setText(QCoreApplication.translate("MainWindow", u"Q num", None))
        self.checkBox_q1d_log_spaced.setText(QCoreApplication.translate("MainWindow", u"Log Spaced", None))
        self.pushButton_1d_measure.setText(QCoreApplication.translate("MainWindow", u"Measure", None))
        self.groupBox_2d_measure.setTitle(QCoreApplication.translate("MainWindow", u"2D", None))
        self.label_q2d_resolution.setText(QCoreApplication.translate("MainWindow", u"Resolution ", None))
        self.label_times.setText(QCoreApplication.translate("MainWindow", u"\u00d7", None))
        self.label_q2d_spacing.setText(QCoreApplication.translate("MainWindow", u"Q spacing", None))
        self.label_q2d_unit.setText(QCoreApplication.translate("MainWindow", u"\u00c5^-1", None))
        self.label_q2d_normal_vector.setText(QCoreApplication.translate("MainWindow", u"Normal vector", None))
        self.checkBox_2d_show_3d.setText(QCoreApplication.translate("MainWindow", u"Show in 3D Q Space", None))
        self.pushButton_2d_measure.setText(QCoreApplication.translate("MainWindow", u"Measure", None))
        self.tabWidget_measure.setTabText(self.tabWidget_measure.indexOf(self.tabPage_direct_measure), QCoreApplication.translate("MainWindow", u"Direct Measure", None))
        self.tabWidget_measure.setTabText(self.tabWidget_measure.indexOf(self.tabPage_virtual_detector), QCoreApplication.translate("MainWindow", u"Virtual Detector", None))
        self.pushButton_scattering.setText(QCoreApplication.translate("MainWindow", u"Virtual Scattering", None))
        self.tabWidget_settings.setTabText(self.tabWidget_settings.indexOf(self.tabPage_scattering), QCoreApplication.translate("MainWindow", u"Scattering Settings", None))
        self.dockWidget_log.setWindowTitle(QCoreApplication.translate("MainWindow", u"Log", None))
    # retranslateUi

