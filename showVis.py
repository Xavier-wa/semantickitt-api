import  subprocess
import time
class VisWindow():
    def __init__(self,script_path,args,) -> None:
        
        self.python_script_path = script_path
        self.args = args

    def __call__(self):
        args_list = [f"-{name}={value}" for name, value in self.args.items()]
        subprocess.Popen(["python", self.python_script_path] + args_list)

dataset = "D:\MyProject\Ridar_SS\Semantic-kitti\dataset"
python_scripts = [
    ["visualize.py",{"d": dataset,
                     "dn":"GT_",
                     "s":"08",}],
    ["visualize.py",{"d":dataset,
                     "dn":"FRNet_",
                     "p":"D:\FileFromRemote\ErrorMap\FRNet",
                     "s":"08",
                     "lg":"True"}],
    ["visualize.py",{"d":dataset,
                     "dn":"SphereFormer_",
                    "p":"D:\FileFromRemote\ErrorMap\SphereFormer",
                     "s":"08",
                     "lg":"True"}],
    ["visualize.py",{"d":dataset,
                     "dn":"PVKD_",
                     "p":"D:\FileFromRemote\ErrorMap\PVKD",
                     "s":"08",
                     "lg":"True"}],

]

for i in python_scripts:
    vis= VisWindow(i[0],i[1])
    vis()
    time.sleep(1)
# subprocess.wait()
# 构造参数列表

# 使用subprocess模块启动另一个Python脚本并传递带名称的参数
