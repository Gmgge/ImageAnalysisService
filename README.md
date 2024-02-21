# 1 ImageAnalysisService

图像分析服务
1) data模块用于存储模型，百度云链接近期放出
2) gradio体验网址 http://120.27.143.171:19192/ 由于服务器配置有限，可能有所卡顿
3) 由于数据集有限，在部分形式印章下可能存在精度较低情况，任何通过审核的真实数据贡献，都可以享受定制化模型训练、训练技巧沟通、GPU部署等协助
4) 推出[数据集分享页面](https://github.com/Gmgge/TrOCR-Seal-Recognition/blob/main/DataSet.md)，包括互联网其他开源链接引用、自制数据集

## 1.1 实现功能
- [x]  web分析api
- [x]  gradio展示
- [x]  印章识别
- [ ]  倾斜图像OCR
- [ ]  推理会话规范初始化
- [ ]  测试用例完善
- [ ]  更精准的模型
- [ ]  whl包构建 更简单化调用
- [ ]  数据开源



## 1.2 环境安装
```
pip install -r requirements.txt
```
## 1.3 运行web分析服务
```
python service_run.py
```

## 1.4 运行gradio服务
```
python service_gradio.py
```

# 2.ImageAnalysisService 开发指南
## 2.1 文件树
```
--| conf 配置文件目录，用于存放可修改的配置文件与项目级别的配置文件
--|--| config.yml 部署时可修改的参数文件
--|--| global_variable.yml 全局变量
--|--| ... 
--| core 核心模块组，用于存放各个分析模块
--|--| __init__.py 显式导入pipeline需要的模块类
--|--|** **分析模块
--|--|--| **config 分析模块自有参数
--|--|--| **api 分析模块自有路由
--|--|--| ... 分析模块自有其他文件
--| libs 通用依赖组件
--| tests 测试模块
--| utils 工具模块
--| web web服务启动模块
```
## 2.2 开发约束
```
1) pipeline 所用到的子类需要集成自相同的基类，以便pipeline模块自动构建与调用
2) core 模块中，分析类定义时，需要设定默认参数，以便调用方可以更简洁的调用
3) core 模块中，自有参数和路由放在该模块文件夹中，以便实现插件式增模块
4) core 模块中，如果逻辑允许，继承base模块中的分析类和参数类，以便实现插件式增模块
5) core 模块中，如果模块需要构建接口，请在模块中构建**api路由文件，以自动注册路由
6) core 模块中，如果存在**api路由文件，需要自动注册相关路由的，设定router = APIRouter(...)，路由注册函数会搜寻该变量名
```


