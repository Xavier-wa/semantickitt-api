import numpy as np
from vispy import app, scene

# 创建 canvas 和 view
canvas = scene.SceneCanvas(keys='interactive', size=(800, 600), show=True)
view = canvas.central_widget.add_view()
view.camera = scene.cameras.TurntableCamera(up='z', fov=60)  # 使用旋转相机

@canvas.events.mouse_press.connect
def on_mouse_press(event):
    # 获取鼠标点击的屏幕坐标
    screen_pos = event.pos
    
    # 转换为归一化设备坐标 (NDC)
    ndc_pos = [(screen_pos[0] / canvas.size[0]) * 2 - 1, 
               ((canvas.size[1] - screen_pos[1]) / canvas.size[1]) * 2 - 1, 
               0]  # z=0 代表近剪切面，z=1 代表远剪切面

    # 使用 unproject 转换屏幕坐标到世界坐标
    world_pos = view.camera.unproject(ndc_pos, viewport=(0, 0, canvas.size[0], canvas.size[1]))
    
    # 输出转换后的世界坐标
    print("World position:", world_pos)

if __name__ == '__main__':
    app.run()
