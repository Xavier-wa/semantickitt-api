import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pdb,yaml
# 创建一个随机数据矩阵
# data = np.random.rand(20, 20)
# 使用seaborn来创建热图
plt.figure(figsize=(30,30))
# data[:,0]=1520
data1 = np.fromfile("D:\FileFromRemote\ErrorMap\FRNet\sequences\\08\ErrorSum\\000000.npy",dtype=np.int32).reshape((20,20))
data2 = np.fromfile("D:\FileFromRemote\ErrorMap\SphereFormer\sequences\\08\ErrorSum\\000000.npy",dtype=np.int32).reshape((20,20))
data3 = np.fromfile("D:\FileFromRemote\ErrorMap\PVKD\sequences\\08\ErrorSum\\000000.npy",dtype=np.int32).reshape((20,20))
# pdb.set_trace()
row_names = ['ig', 'car', 'byc', "mcl","trc","ove","per","bycst","mcst","road","pki","swl","ogd","bui","fen","veg","ruk","ter","pole","tra"]

# 用于演示的多组数据
# data1 = np.random.rand(20, 20)
# data2 = np.random.rand(20, 20)
# data3 = np.random.rand(20, 20)

datasets = [data1, data2, data3]
current_dataset_index = 0

def plot_current_dataset():
    plt.figure(figsize=(8, 6))
    sns.heatmap(datasets[current_dataset_index], annot=True, cmap='coolwarm',fmt='d',xticklabels=row_names,yticklabels=row_names)  # 'annot=True'显示数值，'cmap'定义颜色映射

    plt.title('Matrix Heatmap')
    plt.show()

plot_current_dataset()

def press(event):
    global current_dataset_index
    if event.key == 'z':
        current_dataset_index = (current_dataset_index + 1) % len(datasets)
    elif event.key == 'x':
        current_dataset_index = (current_dataset_index - 1) % len(datasets)
    plot_current_dataset()

plt.gcf().canvas.mpl_connect('key_press_event', press)
plt.show()
