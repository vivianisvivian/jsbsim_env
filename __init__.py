import os
from gym.envs.registration import registry, register, make, spec
from my_jsbsim_env.envs import TASKS  # 存储环境对象
from my_jsbsim_env.catalogs import Catalog

"""

This script registers JSBSimEnv

with OpenAI Gym so that they can be instantiated with a gym.make(id)

 command.


 To use do:

       env = gym.make('GymJsbsim-{task}-v0')

"""

if "JSBSIM_ROOT_DIR" not in os.environ:
    os.environ["JSBSIM_ROOT_DIR"] = os.path.join(os.path.dirname(__file__), "jsbsim")

# 将jsbsim文件放到环境变量中，并起名

for task_name in TASKS:
    # print(task_name)
    register(
        id=f"{task_name}-v0",  # 环境名
        entry_point="my_jsbsim_env.jsbsim_env:JSBSimEnv",
        kwargs=dict(task=TASKS[task_name]),
    )
# 将环境注册到gym中，v0不能少,将任务类传参进去
