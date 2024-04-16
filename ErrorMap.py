import pcl
import open3d as o3d
import yaml
import numpy as np
with open("D:\MyProject\Ridar_SS\semantic-kitti-api\config\semantic-kitti.yaml", 'r') as file:
    semantic = yaml.safe_load(file)
    label_to_color = semantic["color_map"]
import pdb
def getPC(pointCloud,label):
    pc = np.fromfile(pointCloud,dtype=np.float32).reshape((-1,4))
    pdb.set_trace()
    semanticL = np.fromfile(label,dtype=np.int32)& 0xFFFF #label
    # semanticL[semanticL!=90] = 91 #正确为灰色

    point_cloud = o3d.geometry.PointCloud()
    colors = np.array([label_to_color[label] for label in semanticL])
    pdb.set_trace()
    point_cloud.points = o3d.utility.Vector3dVector(pc[:,:-1])  # Extract x, y, z coordinates
    point_cloud.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Normalize remission values to [0, 1]
    return point_cloud

pointfile = "D:\MyProject\Ridar_SS\Semantic-kitti\dataset\sequences\\08\\velodyne\\000010.bin"

labelfile = "D:\FileFromRemote\ErrorMap\SphereFormer\sequences\\08\predictions\\000000.label"
labelfile = "D:\MyProject\Ridar_SS\Semantic-kitti\dataset\sequences\\08\\labels\\000010.label"
pc = getPC(pointfile,labelfile)
o3d.visualization.draw_geometries([pc])