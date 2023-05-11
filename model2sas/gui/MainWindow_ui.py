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
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox,
    QDockWidget, QGridLayout, QGroupBox, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QMainWindow,
    QMdiArea, QMenuBar, QPushButton, QSizePolicy,
    QSpacerItem, QStatusBar, QTabWidget, QTableView,
    QTextBrowser, QTreeView, QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(937, 648)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout_7 = QGridLayout(self.centralwidget)
        self.gridLayout_7.setObjectName(u"gridLayout_7")
        self.mdiArea = QMdiArea(self.centralwidget)
        self.mdiArea.setObjectName(u"mdiArea")
        self.mdiArea.setMinimumSize(QSize(400, 400))

        self.gridLayout_7.addWidget(self.mdiArea, 0, 0, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 937, 22))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.dockWidget_main = QDockWidget(MainWindow)
        self.dockWidget_main.setObjectName(u"dockWidget_main")
        self.dockWidgetContents_3 = QWidget()
        self.dockWidgetContents_3.setObjectName(u"dockWidgetContents_3")
        self.gridLayout_9 = QGridLayout(self.dockWidgetContents_3)
        self.gridLayout_9.setObjectName(u"gridLayout_9")
        self.label_length_unit = QLabel(self.dockWidgetContents_3)
        self.label_length_unit.setObjectName(u"label_length_unit")

        self.gridLayout_9.addWidget(self.label_length_unit, 0, 0, 1, 1)

        self.comboBox_length_unit = QComboBox(self.dockWidgetContents_3)
        self.comboBox_length_unit.addItem("")
        self.comboBox_length_unit.setObjectName(u"comboBox_length_unit")
        self.comboBox_length_unit.setEnabled(False)

        self.gridLayout_9.addWidget(self.comboBox_length_unit, 0, 1, 1, 1)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_9.addItem(self.horizontalSpacer, 0, 2, 1, 1)

        self.label_active_model = QLabel(self.dockWidgetContents_3)
        self.label_active_model.setObjectName(u"label_active_model")
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_active_model.sizePolicy().hasHeightForWidth())
        self.label_active_model.setSizePolicy(sizePolicy)

        self.gridLayout_9.addWidget(self.label_active_model, 0, 3, 1, 1)

        self.groupBox_part = QGroupBox(self.dockWidgetContents_3)
        self.groupBox_part.setObjectName(u"groupBox_part")
        self.gridLayout_5 = QGridLayout(self.groupBox_part)
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.pushButton_build_math_model = QPushButton(self.groupBox_part)
        self.pushButton_build_math_model.setObjectName(u"pushButton_build_math_model")
        self.pushButton_build_math_model.setEnabled(False)

        self.gridLayout_5.addWidget(self.pushButton_build_math_model, 2, 0, 1, 3)

        self.pushButton_part_from_files = QPushButton(self.groupBox_part)
        self.pushButton_part_from_files.setObjectName(u"pushButton_part_from_files")

        self.gridLayout_5.addWidget(self.pushButton_part_from_files, 1, 0, 1, 2)


        self.gridLayout_9.addWidget(self.groupBox_part, 1, 0, 1, 3)

        self.tabWidget_settings = QTabWidget(self.dockWidgetContents_3)
        self.tabWidget_settings.setObjectName(u"tabWidget_settings")
        sizePolicy1 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.tabWidget_settings.sizePolicy().hasHeightForWidth())
        self.tabWidget_settings.setSizePolicy(sizePolicy1)
        self.tabPage_model = QWidget()
        self.tabPage_model.setObjectName(u"tabPage_model")
        self.gridLayout_8 = QGridLayout(self.tabPage_model)
        self.gridLayout_8.setObjectName(u"gridLayout_8")
        self.label_model_params = QLabel(self.tabPage_model)
        self.label_model_params.setObjectName(u"label_model_params")

        self.gridLayout_8.addWidget(self.label_model_params, 2, 0, 1, 2)

        self.label_part_device = QLabel(self.tabPage_model)
        self.label_part_device.setObjectName(u"label_part_device")

        self.gridLayout_8.addWidget(self.label_part_device, 0, 0, 1, 1)

        self.tableView_model_params = QTableView(self.tabPage_model)
        self.tableView_model_params.setObjectName(u"tableView_model_params")
        self.tableView_model_params.setEnabled(False)
        sizePolicy2 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.tableView_model_params.sizePolicy().hasHeightForWidth())
        self.tableView_model_params.setSizePolicy(sizePolicy2)

        self.gridLayout_8.addWidget(self.tableView_model_params, 3, 0, 1, 4)

        self.groupBox_transform = QGroupBox(self.tabPage_model)
        self.groupBox_transform.setObjectName(u"groupBox_transform")
        self.groupBox_transform.setEnabled(False)
        self.gridLayout_4 = QGridLayout(self.groupBox_transform)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.pushButton_delete_transform = QPushButton(self.groupBox_transform)
        self.pushButton_delete_transform.setObjectName(u"pushButton_delete_transform")

        self.gridLayout_4.addWidget(self.pushButton_delete_transform, 0, 2, 1, 1)

        self.comboBox_transform_type = QComboBox(self.groupBox_transform)
        self.comboBox_transform_type.addItem("")
        self.comboBox_transform_type.addItem("")
        self.comboBox_transform_type.setObjectName(u"comboBox_transform_type")

        self.gridLayout_4.addWidget(self.comboBox_transform_type, 0, 0, 1, 1)

        self.pushButton_add_transform = QPushButton(self.groupBox_transform)
        self.pushButton_add_transform.setObjectName(u"pushButton_add_transform")

        self.gridLayout_4.addWidget(self.pushButton_add_transform, 0, 1, 1, 1)

        self.pushButton_apply_transform = QPushButton(self.groupBox_transform)
        self.pushButton_apply_transform.setObjectName(u"pushButton_apply_transform")

        self.gridLayout_4.addWidget(self.pushButton_apply_transform, 5, 2, 1, 1)

        self.tableView_transform = QTableView(self.groupBox_transform)
        self.tableView_transform.setObjectName(u"tableView_transform")
        sizePolicy2.setHeightForWidth(self.tableView_transform.sizePolicy().hasHeightForWidth())
        self.tableView_transform.setSizePolicy(sizePolicy2)

        self.gridLayout_4.addWidget(self.tableView_transform, 4, 0, 1, 3)


        self.gridLayout_8.addWidget(self.groupBox_transform, 5, 0, 1, 4)

        self.lineEdit_real_lattice_1d_size = QLineEdit(self.tabPage_model)
        self.lineEdit_real_lattice_1d_size.setObjectName(u"lineEdit_real_lattice_1d_size")

        self.gridLayout_8.addWidget(self.lineEdit_real_lattice_1d_size, 4, 2, 1, 1)

        self.label_real_lattice_1d_size = QLabel(self.tabPage_model)
        self.label_real_lattice_1d_size.setObjectName(u"label_real_lattice_1d_size")

        self.gridLayout_8.addWidget(self.label_real_lattice_1d_size, 4, 0, 1, 2)

        self.pushButton_sampling = QPushButton(self.tabPage_model)
        self.pushButton_sampling.setObjectName(u"pushButton_sampling")
        self.pushButton_sampling.setEnabled(False)

        self.gridLayout_8.addWidget(self.pushButton_sampling, 4, 3, 1, 1)

        self.checkBox_volume = QCheckBox(self.tabPage_model)
        self.checkBox_volume.setObjectName(u"checkBox_volume")
        self.checkBox_volume.setEnabled(False)

        self.gridLayout_8.addWidget(self.checkBox_volume, 6, 1, 1, 1)

        self.pushButton_plot_model = QPushButton(self.tabPage_model)
        self.pushButton_plot_model.setObjectName(u"pushButton_plot_model")
        self.pushButton_plot_model.setEnabled(False)

        self.gridLayout_8.addWidget(self.pushButton_plot_model, 6, 2, 1, 2)

        self.checkBox_voxel = QCheckBox(self.tabPage_model)
        self.checkBox_voxel.setObjectName(u"checkBox_voxel")
        self.checkBox_voxel.setEnabled(False)

        self.gridLayout_8.addWidget(self.checkBox_voxel, 6, 0, 1, 1)

        self.comboBox_model_device = QComboBox(self.tabPage_model)
        self.comboBox_model_device.addItem("")
        self.comboBox_model_device.setObjectName(u"comboBox_model_device")
        self.comboBox_model_device.setEnabled(False)

        self.gridLayout_8.addWidget(self.comboBox_model_device, 0, 1, 1, 1)

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
        self.lineEdit_reciprocal_lattice_1d_size.setEnabled(False)

        self.gridLayout_3.addWidget(self.lineEdit_reciprocal_lattice_1d_size, 0, 1, 1, 1)

        self.tabWidget_measure = QTabWidget(self.tabPage_scattering)
        self.tabWidget_measure.setObjectName(u"tabWidget_measure")
        self.tabWidget_measure.setEnabled(True)
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
        self.pushButton_1d_measure.setEnabled(False)

        self.verticalLayout.addWidget(self.pushButton_1d_measure)


        self.verticalLayout_3.addWidget(self.groupBox_1d_measure)

        self.groupBox_2d_measure = QGroupBox(self.tabPage_direct_measure)
        self.groupBox_2d_measure.setObjectName(u"groupBox_2d_measure")
        self.groupBox_2d_measure.setEnabled(False)
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
        self.pushButton_scattering.setEnabled(False)

        self.gridLayout_3.addWidget(self.pushButton_scattering, 1, 0, 1, 2)

        self.tabWidget_settings.addTab(self.tabPage_scattering, "")

        self.gridLayout_9.addWidget(self.tabWidget_settings, 1, 3, 3, 1)

        self.groupBox_assembly = QGroupBox(self.dockWidgetContents_3)
        self.groupBox_assembly.setObjectName(u"groupBox_assembly")
        self.gridLayout_6 = QGridLayout(self.groupBox_assembly)
        self.gridLayout_6.setObjectName(u"gridLayout_6")
        self.comboBox_assmbly_list = QComboBox(self.groupBox_assembly)
        self.comboBox_assmbly_list.setObjectName(u"comboBox_assmbly_list")
        self.comboBox_assmbly_list.setEnabled(False)

        self.gridLayout_6.addWidget(self.comboBox_assmbly_list, 3, 2, 1, 1)

        self.pushButton_new_assembly = QPushButton(self.groupBox_assembly)
        self.pushButton_new_assembly.setObjectName(u"pushButton_new_assembly")
        self.pushButton_new_assembly.setEnabled(False)

        self.gridLayout_6.addWidget(self.pushButton_new_assembly, 0, 0, 1, 2)

        self.comboBox_assembly_device = QComboBox(self.groupBox_assembly)
        self.comboBox_assembly_device.addItem("")
        self.comboBox_assembly_device.setObjectName(u"comboBox_assembly_device")
        self.comboBox_assembly_device.setEnabled(False)

        self.gridLayout_6.addWidget(self.comboBox_assembly_device, 0, 2, 1, 1)

        self.pushButton_add_to_assembly = QPushButton(self.groupBox_assembly)
        self.pushButton_add_to_assembly.setObjectName(u"pushButton_add_to_assembly")
        self.pushButton_add_to_assembly.setEnabled(False)

        self.gridLayout_6.addWidget(self.pushButton_add_to_assembly, 3, 0, 1, 2)


        self.gridLayout_9.addWidget(self.groupBox_assembly, 2, 0, 1, 3)

        self.treeView_models = QTreeView(self.dockWidgetContents_3)
        self.treeView_models.setObjectName(u"treeView_models")
        sizePolicy2.setHeightForWidth(self.treeView_models.sizePolicy().hasHeightForWidth())
        self.treeView_models.setSizePolicy(sizePolicy2)
        self.treeView_models.setMinimumSize(QSize(100, 0))
        self.treeView_models.setEditTriggers(QAbstractItemView.NoEditTriggers)

        self.gridLayout_9.addWidget(self.treeView_models, 3, 0, 1, 3)

        self.dockWidget_main.setWidget(self.dockWidgetContents_3)
        MainWindow.addDockWidget(Qt.LeftDockWidgetArea, self.dockWidget_main)
        self.dockWidget_log = QDockWidget(MainWindow)
        self.dockWidget_log.setObjectName(u"dockWidget_log")
        self.dockWidgetContents_4 = QWidget()
        self.dockWidgetContents_4.setObjectName(u"dockWidgetContents_4")
        self.verticalLayout_4 = QVBoxLayout(self.dockWidgetContents_4)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.textBrowser_log = QTextBrowser(self.dockWidgetContents_4)
        self.textBrowser_log.setObjectName(u"textBrowser_log")
        self.textBrowser_log.setEnabled(True)

        self.verticalLayout_4.addWidget(self.textBrowser_log)

        self.dockWidget_log.setWidget(self.dockWidgetContents_4)
        MainWindow.addDockWidget(Qt.LeftDockWidgetArea, self.dockWidget_log)
        QWidget.setTabOrder(self.pushButton_part_from_files, self.pushButton_build_math_model)
        QWidget.setTabOrder(self.pushButton_build_math_model, self.comboBox_transform_type)
        QWidget.setTabOrder(self.comboBox_transform_type, self.tableView_transform)
        QWidget.setTabOrder(self.tableView_transform, self.lineEdit_real_lattice_1d_size)
        QWidget.setTabOrder(self.lineEdit_real_lattice_1d_size, self.pushButton_plot_model)
        QWidget.setTabOrder(self.pushButton_plot_model, self.tableView_model_params)
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
        QWidget.setTabOrder(self.pushButton_2d_measure, self.pushButton_new_assembly)
        QWidget.setTabOrder(self.pushButton_new_assembly, self.pushButton_add_to_assembly)

        self.retranslateUi(MainWindow)
        self.pushButton_build_math_model.clicked.connect(MainWindow.build_math_model)
        self.pushButton_part_from_files.clicked.connect(MainWindow.load_model_files)
        self.treeView_models.clicked.connect(MainWindow.selected_model_settings)
        self.pushButton_sampling.clicked.connect(MainWindow.sampling)
        self.pushButton_plot_model.clicked.connect(MainWindow.plot_model)
        self.pushButton_scattering.clicked.connect(MainWindow.virtual_scattering)
        self.pushButton_1d_measure.clicked.connect(MainWindow.measure_1d)

        self.tabWidget_settings.setCurrentIndex(0)
        self.tabWidget_measure.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Model2SAS", None))
        self.dockWidget_main.setWindowTitle(QCoreApplication.translate("MainWindow", u"Main", None))
        self.label_length_unit.setText(QCoreApplication.translate("MainWindow", u"Length Unit", None))
        self.comboBox_length_unit.setItemText(0, QCoreApplication.translate("MainWindow", u"\u00c5", None))

        self.label_active_model.setText(QCoreApplication.translate("MainWindow", u"No Active Model", None))
        self.groupBox_part.setTitle(QCoreApplication.translate("MainWindow", u"Part Model", None))
        self.pushButton_build_math_model.setText(QCoreApplication.translate("MainWindow", u"Build Math Model", None))
        self.pushButton_part_from_files.setText(QCoreApplication.translate("MainWindow", u"Load From File(s)", None))
        self.label_model_params.setText(QCoreApplication.translate("MainWindow", u"Model Parameters", None))
        self.label_part_device.setText(QCoreApplication.translate("MainWindow", u"Device", None))
        self.groupBox_transform.setTitle(QCoreApplication.translate("MainWindow", u"Transform", None))
        self.pushButton_delete_transform.setText(QCoreApplication.translate("MainWindow", u"Delete", None))
        self.comboBox_transform_type.setItemText(0, QCoreApplication.translate("MainWindow", u"Translate", None))
        self.comboBox_transform_type.setItemText(1, QCoreApplication.translate("MainWindow", u"Rotate", None))

        self.pushButton_add_transform.setText(QCoreApplication.translate("MainWindow", u"Add", None))
        self.pushButton_apply_transform.setText(QCoreApplication.translate("MainWindow", u"Apply", None))
        self.lineEdit_real_lattice_1d_size.setText("")
        self.label_real_lattice_1d_size.setText(QCoreApplication.translate("MainWindow", u"Points in longest edge", None))
        self.pushButton_sampling.setText(QCoreApplication.translate("MainWindow", u"Sampling", None))
        self.checkBox_volume.setText(QCoreApplication.translate("MainWindow", u"Volume", None))
        self.pushButton_plot_model.setText(QCoreApplication.translate("MainWindow", u"Plot", None))
        self.checkBox_voxel.setText(QCoreApplication.translate("MainWindow", u"Voxel", None))
        self.comboBox_model_device.setItemText(0, QCoreApplication.translate("MainWindow", u"cpu", None))

        self.tabWidget_settings.setTabText(self.tabWidget_settings.indexOf(self.tabPage_model), QCoreApplication.translate("MainWindow", u"Model Settings", None))
        self.label_reciprocal_lattice_1d_size.setText(QCoreApplication.translate("MainWindow", u"Edge points num in reciprocal grid", None))
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
        self.groupBox_assembly.setTitle(QCoreApplication.translate("MainWindow", u"Assembly Model", None))
        self.pushButton_new_assembly.setText(QCoreApplication.translate("MainWindow", u"New Assembly", None))
        self.comboBox_assembly_device.setItemText(0, QCoreApplication.translate("MainWindow", u"cpu", None))

        self.pushButton_add_to_assembly.setText(QCoreApplication.translate("MainWindow", u"Add to Assembly", None))
        self.dockWidget_log.setWindowTitle(QCoreApplication.translate("MainWindow", u"Log", None))
    # retranslateUi

