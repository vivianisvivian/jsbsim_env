from os import listdir
from os import path
import importlib


dir_path = path.dirname(path.realpath(__file__))
# __file__是当前绝对路径 代码形式/
# path.realpath(__file__) 是当前绝对路径，文件形式\
# path.dirname(path.realpath(__file__)) 返回的是父路径

ENV_file_names = [f for f in listdir(dir_path) if path.isfile(path.join(dir_path, f))]
# 得到 ENV_file_names(list,列表类型）, 存储dir_path下所有文件名字（除去文件夹）

TASKS_NAMES = {}
for f in ENV_file_names:
    name_file = f.split(".")[0]
    if name_file[-4:] == "task":
        name_class = name_file.title().replace("_", "")  # 改写名
        TASKS_NAMES[name_file] = name_class
# 得到TASKS_NAMES(dict,字典类型）, 文件名对应的环境名, 环境名改大写首字母，去掉下划线

TASKS = {}
for f in TASKS_NAMES:
    module = importlib.import_module("my_jsbsim_env.envs." + f)  # 类，创建对象
    my_class = getattr(module, TASKS_NAMES[f])  # 获取对象中的属性值
    TASKS[TASKS_NAMES[f]] = my_class
# 得到TASKS(dict,字典类型)，环境名对应的类位置