# 该文件用于存放多模块之间共享的变量 请注意共享变量的使用，包括串行时的读写顺序，并行的读写安全等
# module_pools 与 image_analysis_pipeline 中可能包含共同的分析模块，当前设计原因为：
# 1.当前两个全局变量功能不同，module_pools为了模块之间存在依赖时能够调度到其他模块，image_analysis_pipeline为了自动化分析请求任务

from conf.service_args import service_config

# 参考官方文档实现 https://docs.python.org/3/faq/programming.html#how-do-i-share-global-variables-across-modules
# 全局模块池，用于模块互相依赖时，从该模块指针池中调用所需要的模块
module_pools = {one_module: None for one_module in service_config["ModuleSwitch"].keys()}

# 分析pipeline 初始化为空
image_analysis_pipeline = None

if __name__ == "__main__":
    print(module_pools)
