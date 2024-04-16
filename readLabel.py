import numpy as np

an = np.fromfile("D:\\MyProject\\Ridar_SS\\semantic-kitti-api\\rangeView\\STR\\dataset\\sequences\\11\\velodyne\\000001.bin",dtype=np.float32).reshape((-1,4))
print(an.shape)
ab = np.fromfile("D:\\MyProject\\Ridar_SS\\semantic-kitti-api\\rangeView\\STR\\dataset\\sequences\\00\\labels\\000000.label",dtype=np.uint32)
print(ab)