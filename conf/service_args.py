import yaml
import os
from utils.conf_utils import get_open_modules

# 服务名称
service_name = "ImageAnalysis"
# 项目根目录 需要打包成可执行程序时，请切换成"."
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# 路由根目录
router_file_root = os.path.join(project_root, "core")
# 读取yml配置
config_yml_path = os.path.join(project_root, "conf/config.yml")
service_config = yaml.load(open(config_yml_path, "rb"), Loader=yaml.Loader)
# 启动模块列表
open_models = get_open_modules(service_config["ModuleSwitch"])


if __name__ == "__main__":
    print(project_root)
