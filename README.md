# Model2SAS
Program to generate small angle scattering curve from 3D model.

开发分为两个部分，一是实现核心功能的库Model2SAS，二是为了使用方便编写的GUI。

## 运行环境

Model2SAS库所依赖的环境：

- Python 3.8.5
- numpy 1.19.2
- scipy 1.5.2
- matplotlib 3.3.2,
- numpy-stl 2.13.0
- psutil 5.7.2

除以上依赖的库，GUI还需要：

- PyQt5 5.15.2

如果需要使用GPU加速，则还需要：

- PyTorch 1.9.0

注：使用GPU加速功能请直接使用源码，打包的版本未提供此功能。

以上是开发所使用的环境，并非运行必须。总的来说并没有用到很多新版本才加入的功能，所以对各个库的版本要求并不严格。

