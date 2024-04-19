import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data1 = np.fromfile(r"D:\FileFromRemote\ErrorMap\FRNet\sequences\08\ErrorSum\000000.npy", dtype=np.int32).reshape((20, 20))
data2 = np.fromfile(r"D:\FileFromRemote\ErrorMap\SphereFormer\sequences\08\ErrorSum\000000.npy", dtype=np.int32).reshape((20, 20))
data3 = np.fromfile(r"D:\FileFromRemote\ErrorMap\PVKD\sequences\08\ErrorSum\000000.npy", dtype=np.int32).reshape((20, 20))
dataset1 = [data1,data2,data3]
# 假设 data1, data2, data3 已经加载
datasets = [dataset1, dataset1, dataset1]
current_heatmap_index = 0

# 创建一个具有不同布局的图形窗口
fig = plt.figure(figsize=(15, 10))
# 定义子图的网格布局
ax1 = plt.subplot2grid((2, 2), (0, 0))
ax2 = plt.subplot2grid((2, 2), (0, 1))
ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)

axs = [ax1, ax2, ax3]  # 存储轴对象的列表

def plot_heatmaps(index):
    for i, ax in enumerate(axs):
        ax.clear()  # 清除当前子图内容
        sns.heatmap(datasets[i][index], annot=True, cmap='coolwarm', fmt='d',
                    cbar=False, ax=ax)  # 绘制热图
        ax.set_title(f'Dataset {i+1} - Heatmap {index+1}')
    plt.tight_layout()  # 自动调整子图参数，以给定的填充

plot_heatmaps(current_heatmap_index)  # 初次绘图

def press(event):
    global current_heatmap_index
    if event.key == 'right':  # 按右箭头键增加索引
        current_heatmap_index = (current_heatmap_index + 1) % len(datasets[0])
    elif event.key == 'left':  # 按左箭头键减少索引
        current_heatmap_index = (current_heatmap_index - 1) % len(datasets[0])
    plot_heatmaps(current_heatmap_index)
    fig.canvas.draw() 
