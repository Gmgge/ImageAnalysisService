import re
import os
import yaml

from conf.service_args import project_root


def read_yaml(yaml_path):
    with open(yaml_path, "rb") as f:
        data = yaml.load(f, Loader=yaml.Loader)
    return data


def check_dirs_in_path(dirs_name, file_path):
    """
    判断file_path是否包含dirs_name中任意一个文件夹，如果包含则分析器文件夹名，否则返回空字符串
    """
    hit_dir = ""
    for dir_name in dirs_name:
        if dir_name in file_path:
            hit_dir = dir_name
            break
    return hit_dir


def get_import_path(file_path, root=project_root):
    """
    根据待导入py文件路径与项目根路径，获取导入路径
    限制规则：file_path必须为root目录下的文件
    导入路径类型为{}.{}.{}
    """
    import_path = ""  # 默认为空路径
    # 检查路径
    if os.path.isfile(file_path) and os.path.isdir(root) and len(file_path) > (len(root)):
        path_split = r'[\\|/]'
        relative_path = os.path.relpath(file_path, root)
        path_list = re.split(path_split, relative_path)
        # 前缀路径组合
        import_path = path_list[0]
        for one_part in path_list[1:-1]:
            import_path = "{}.{}".format(import_path, one_part)
        # 后缀文件组合
        if len(path_list) == 1:
            import_path = os.path.splitext(path_list[0])[0]
        else:
            import_path = "{}.{}".format(import_path, os.path.splitext(path_list[-1])[0])
    return import_path


if __name__ == "__main__":
    test_path = r"D:\GitHubWorkSpace\Media-Analysis-Services\core\analysis_pipeline\image_analysis_pipeline_api.py"
    get_import_path(test_path)
