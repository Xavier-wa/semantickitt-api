#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import vispy
from vispy.scene import visuals, SceneCanvas
import numpy as np
from matplotlib import pyplot as plt
import vispy.scene
from auxiliary.laserscan import LaserScan, SemLaserScan
import os 
import pdb

def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        filenames.sort()
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))

class ErrorMapVis:
  """Class that creates and handles a visualizer for a pointcloud"""

  def __init__(self, scan,scan_names, label_names, offset=0,
               semantics=True, instances=False, images=True, link=False,dir="",class_dict=None,scan_pre=None,scanes=None,scanes_names=None):
    self.scan = scan
    self.scan_names = scan_names
    self.label_names = label_names
    self.offset = offset
    self.total = len(self.scan_names)
    self.semantics = semantics
    self.instances = instances
    self.images = images
    self.link = link
    self.dir = dir
    self.class_dict = class_dict
    self.scan_pre = scan_pre
    self.scanes = scanes
    self.scanes_names = scanes_names
    self.prediction_path =[[],[],[]]
    self.errormap_path =[[],[],[]]
    # pdb.set_trace()
    for i in range(len(self.scanes_names)):
        self.prediction_path[i] += absoluteFilePaths(self.scanes_names[i]+"\\sequences\\08\labels\\")
        self.errormap_path[i] += absoluteFilePaths(self.scanes_names[i]+"\\sequences\\08\predictions\\") 
    # pdb.set_trace()
    # sanity check
    if not self.semantics and self.instances:
      print("Instances are only allowed in when semantics=True")
      raise ValueError

    self.reset()
    self.update_scan()

  def reset(self):
    """ Reset. """
    # last key press (it should have a mutex, but visualization is not
    # safety critical, so let's do things wrong)
    self.action = "no"  # no, next, back, quit are the possibilities

    # new canvas prepared for visualizing data
    self.canvas = SceneCanvas(keys='interactive', show=True)
    # interface (n next, b back, q quit, very simple)
    self.canvas.events.key_press.connect(self.key_press)
    self.canvas.events.draw.connect(self.draw)
    # grid
    self.grid = self.canvas.central_widget.add_grid()
    self.text = visuals.Text('',color='white',pos=(20,20),parent=self.canvas.scene)
    # laserscan part
    self.scan_view = vispy.scene.widgets.ViewBox(
        border_color='white', parent=self.canvas.scene)
    self.grid.add_widget(self.scan_view, 0, 0)
    self.scan_vis = visuals.Markers()
    self.scan_view.camera = 'turntable'
    self.scan_view.add(self.scan_vis)
    visuals.XYZAxis(parent=self.scan_view.scene)

    #Xavier 
    model_name = ["Frnet","Pvkd","Sphere"]
    pred_grid_index = [(0,2),(1,0),(1,2)]
    error_grid_index = [(0,3),(1,1),(1,3)]
    # predictuib 
    self.predict_viewes = [vispy.scene.widgets.ViewBox(
        border_color='white', parent=self.canvas.scene
    ) for i in range(3)]
    self.predict_vis = [visuals.Markers() for i in range(3)]

    self.errormap_viewes = [vispy.scene.widgets.ViewBox(
        border_color='white', parent=self.canvas.scene
    ) for i in range(3)]
    self.errormap_vis = [visuals.Markers() for i in range(3)]

    for i in range(len(self.predict_viewes)):
        self.grid.add_widget(self.predict_viewes[i],pred_grid_index[i][0],pred_grid_index[i][1])
        self.predict_viewes[i].camera = 'turntable'
        self.predict_viewes[i]
        self.predict_viewes[i].add(self.predict_vis[i])
        self.predict_viewes[i].add(visuals.Text(model_name[i]+f'_pred',pos=((0,0,2)), color='white',font_size=148))
        visuals.XYZAxis(parent=self.predict_viewes[i].scene)

    for i in range(len(self.errormap_viewes)):
        self.grid.add_widget(self.errormap_viewes[i],error_grid_index[i][0],error_grid_index[i][1])
        self.errormap_viewes[i].camera = 'turntable'
        self.errormap_viewes[i].add(self.errormap_vis[i])
        self.errormap_viewes[i].add(visuals.Text(model_name[i]+f'_errmap',pos=((0,0,2)), color='white',font_size=148))
        visuals.XYZAxis(parent=self.errormap_viewes[i].scene)

    


    if self.class_dict != None:
      pass
    # add semantics
    if self.semantics:
      print("Using semantics in visualizer")
      self.sem_view = vispy.scene.widgets.ViewBox(
          border_color='white', parent=self.canvas.scene)
      self.grid.add_widget(self.sem_view, 0, 1)
      self.sem_vis = visuals.Markers()
      test = visuals.Text("Hello, World!", pos=((0,0,2)), color='white',font_size=148)
      self.sem_view.camera = 'turntable'
      self.sem_view.add(test)
      self.sem_view.add(self.sem_vis)
      visuals.XYZAxis(parent=self.sem_view.scene)
      if self.link:
        self.sem_view.camera.link(self.scan_view.camera)

    if self.instances:
      print("Using instances in visualizer")
      self.inst_view = vispy.scene.widgets.ViewBox(
          border_color='white', parent=self.canvas.scene)
      self.grid.add_widget(self.inst_view, 0, 2)
      self.inst_vis = visuals.Markers()
      self.inst_view.camera = 'turntable'
      self.inst_view.add(self.inst_vis)
      visuals.XYZAxis(parent=self.inst_view.scene)
      if self.link:
        self.inst_view.camera.link(self.scan_view.camera)

    # add a view for the depth
    if self.images:
      # img canvas size
      self.multiplier = 1
      self.canvas_W = 1024
      self.canvas_H = 64
      if self.semantics:
        self.multiplier += 1
      if self.instances:
        self.multiplier += 1

      # new canvas for img
      self.img_canvas = SceneCanvas(keys='interactive', show=True,
                                    size=(self.canvas_W, self.canvas_H * self.multiplier))
      # grid
      self.img_grid = self.img_canvas.central_widget.add_grid()
      # interface (n next, b back, q quit, very simple)
      self.img_canvas.events.key_press.connect(self.key_press)
      self.img_canvas.events.draw.connect(self.draw)
      self.img_view = vispy.scene.widgets.ViewBox(
          border_color='white', parent=self.img_canvas.scene)
      self.img_grid.add_widget(self.img_view, 0, 0)
      self.img_vis = visuals.Image(cmap='viridis')
      self.img_view.add(self.img_vis)

      # add image semantics
      if self.semantics:
        self.sem_img_view = vispy.scene.widgets.ViewBox(
            border_color='white', parent=self.img_canvas.scene)
        self.img_grid.add_widget(self.sem_img_view, 1, 0)
        self.sem_img_vis = visuals.Image(cmap='viridis')
        self.sem_img_view.add(self.sem_img_vis)

    # add instances
    if self.instances:
      self.inst_img_view = vispy.scene.widgets.ViewBox(
          border_color='white', parent=self.img_canvas.scene)
      self.img_grid.add_widget(self.inst_img_view, 2, 0)
      self.inst_img_vis = visuals.Image(cmap='viridis')
      self.inst_img_view.add(self.inst_img_vis)
      if self.link:
        self.inst_view.camera.link(self.scan_view.camera)

  def get_mpl_colormap(self, cmap_name):
    cmap = plt.get_cmap(cmap_name)

    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)

    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]

    return color_range.reshape(256, 3).astype(np.float32) / 255.0
  
  def update_scan(self):
    # first open data
    self.scan.open_scan(self.scan_names[self.offset])

    for i in range(len(self.scanes_names)):
        self.scanes[i].open_scan(self.scan_names[self.offset])
        # pdb.set_trace()
        if self.semantics:
            self.scanes[i].open_label(self.prediction_path[i][self.offset])
            self.scanes[i].colorize()
        self.predict_vis[i].set_data(
            self.scanes[i].points,
            face_color=self.scanes[i].sem_label_color[..., ::-1],
            edge_color=self.scanes[i].sem_label_color[..., ::-1],
            size=1)
    
    for i in range(len(self.scanes_names)):
        self.scanes[i].open_scan(self.scan_names[self.offset])
        if self.semantics:
            self.scanes[i].open_label(self.errormap_path[i][self.offset])
            self.scanes[i].colorize()
        self.errormap_vis[i].set_data(
            self.scanes[i].points,
            face_color=self.scanes[i].sem_label_color[..., ::-1],
            edge_color=self.scanes[i].sem_label_color[..., ::-1],
            size=1)

    if self.semantics:
      self.scan.open_label(self.label_names[self.offset])
      self.scan.colorize()
      if self.scan_pre != None:
        # import pdb
        # pdb.set_trace()
        self.scan_pre.open_scan(self.scan_names[self.offset])
        self.scan_pre.open_label(self.label_names[self.offset].replace("predictions","labels"))
        self.scan_pre.colorize()
    # import pdb
    # pdb.set_trace()
    # then change names
    title = self.dir+"scan " + str(self.offset)
    self.canvas.title = title
    if self.images:
      self.img_canvas.title = title

    # then do all the point cloud stuff

    # plot scan
    power = 16
    # print()
    range_data = np.copy(self.scan.unproj_range)
    # print(range_data.max(), range_data.min())
    range_data = range_data**(1 / power)
    # print(range_data.max(), range_data.min())
    viridis_range = ((range_data - range_data.min()) /
                     (range_data.max() - range_data.min()) *
                     255).astype(np.uint8)
    viridis_map = self.get_mpl_colormap("viridis")
    viridis_colors = viridis_map[viridis_range]
    if self.scan_pre != None:
      self.scan_vis.set_data(self.scan.points,
                             face_color=self.scan_pre.sem_label_color[..., ::-1],
                            edge_color=self.scan_pre.sem_label_color[..., ::-1],
                            size=1)
    else:
      self.scan_vis.set_data(self.scan.points,
                            face_color=viridis_colors[..., ::-1],
                            edge_color=viridis_colors[..., ::-1],
                            size=1)

    # plot semantics
    if self.semantics:
      self.sem_vis.set_data(self.scan.points,
                            face_color=self.scan.sem_label_color[..., ::-1],
                            edge_color=self.scan.sem_label_color[..., ::-1],
                            size=1)

    # plot instances
    if self.instances:
      self.inst_vis.set_data(self.scan.points,
                             face_color=self.scan.inst_label_color[..., ::-1],
                             edge_color=self.scan.inst_label_color[..., ::-1],
                             size=1)

    if self.images:
      # now do all the range image stuff
      # plot range image
      data = np.copy(self.scan.proj_range)
      # print(data[data > 0].max(), data[data > 0].min())
      data[data > 0] = data[data > 0]**(1 / power)
      data[data < 0] = data[data > 0].min()
      # print(data.max(), data.min())
      data = (data - data[data > 0].min()) / \
          (data.max() - data[data > 0].min())
      # print(data.max(), data.min())
      self.img_vis.set_data(data)
      self.img_vis.update()

      if self.semantics:
        self.sem_img_vis.set_data(self.scan.proj_sem_color[..., ::-1])
        self.sem_img_vis.update()

      if self.instances:
        self.inst_img_vis.set_data(self.scan.proj_inst_color[..., ::-1])
        self.inst_img_vis.update()

  # interface
  def key_press(self, event):
    self.canvas.events.key_press.block()
    if self.images:
      self.img_canvas.events.key_press.block()
    if event.key == 'N':
      self.offset += 1
      if self.offset >= self.total:
        self.offset = 0
      self.update_scan()
    elif event.key == 'B':
      self.offset -= 1
      if self.offset < 0:
        self.offset = self.total - 1
      self.update_scan()
    elif event.key == 'Q' or event.key == 'Escape':
      self.destroy()

  def draw(self, event):
    if self.canvas.events.key_press.blocked():
      self.canvas.events.key_press.unblock()
    if self.images and self.img_canvas.events.key_press.blocked():
      self.img_canvas.events.key_press.unblock()

  def moveMouse(self,event):
    global last_selected_point
    if event.button == 1 and event.is_dragging:
        # 如果左键被按下且鼠标在拖动，移动摄像头
        view.camera.drag_to(event.tr.pos, event.tr.last_event.pos)
    else:
        # 在这里确定鼠标当前悬停的点
        # 使用 event.pos 获取当前鼠标位置
        pos = event.pos
        # 在这里你需要实现一个函数来找到最近的点
        # 例如，你可以使用 NumPy 的几何函数
        # last_selected_point = find_nearest_point(pos, points)
        # 假设我们有一个找到的点的位置
        nearest_point_pos = np.random.normal(size=(3,))  # 这里应是实际逻辑的替代
        # 更新文本标签的内容和位置
        text.text = f'({nearest_point_pos[0]:.2f}, {nearest_point_pos[1]:.2f}, {nearest_point_pos[2]:.2f})'
        text.pos = pos
        text.update()

  def destroy(self):
    # destroy the visualization
    self.canvas.close()
    if self.images:
      self.img_canvas.close()
    vispy.app.quit()

  def run(self):
    vispy.app.run()
