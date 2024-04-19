import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')  # 使用TkAgg后端
import matplotlib.pyplot as plt
import os
row_names = ['ig', 'car', 'byc', "mcl", "trc", "ove", "per", "bycst", "mcst", "road", "pki", "swl", "ogd", "bui", "fen", "veg", "ruk", "ter", "pole", "tra"]

def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        filenames.sort()
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))

basePath = r"D:\FileFromRemote\ErrorMap\model\sequences\08\ErrorSum"
fr_idx,pvkd_idx,sphere_idx = [],[],[]
fr_idx += absoluteFilePaths(basePath.replace("model","FRNet"))
pvkd_idx += absoluteFilePaths(basePath.replace("model","PVKD"))
sphere_idx += absoluteFilePaths(basePath.replace("model","SphereFormer"))



datasets = [fr_idx, pvkd_idx, sphere_idx]
dataset_name = ['fr_idx','pvkd_idx','sphere_idx']
current_heatmap_index = 0

# 初始化4个热图的绘图
fig, axs = plt.subplots(2, 2, figsize=(10, 10))  # 创建一个2x2的子图布局
plt.tight_layout()

def plot_heatmaps(index):
    # 以2x2的布局展示每个datasets的第index个热图数据
    for i, dataset in enumerate(datasets):

        ax = axs[i//2, i%2]  # 定位到适当的子图
        ax.clear()  # 清除当前子图内容
        data=np.fromfile(dataset[index],dtype=np.int32).reshape((20,20))
        sns.heatmap(data, annot=True, cmap='coolwarm', fmt='d',
                    cbar=False, ax=ax,xticklabels=row_names,yticklabels=row_names)  # 绘制热图
        ax.set_title(f'{dataset_name[i]}-{index}')

plot_heatmaps(current_heatmap_index)  # 初次绘图

def press(event):
    global current_heatmap_index
    if event.key == 'right':  # 按右箭头键增加索引
        current_heatmap_index = (current_heatmap_index + 1) % len(datasets[0])
    elif event.key == 'left':  # 按左箭头键减少索引
        current_heatmap_index = (current_heatmap_index - 1) % len(datasets[0])
    plot_heatmaps(current_heatmap_index)  # 更新热图
    fig.canvas.draw()  # 重绘整个图形

# 连接按键事件
fig.canvas.mpl_connect('key_press_event', press)
plt.show()  # 显