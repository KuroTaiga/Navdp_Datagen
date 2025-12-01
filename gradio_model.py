import numpy as np

class BicycleModel:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.v = 0.0
        self.heading_theta = 0.0

    def predict(self, dt, control):
        a, delta = control
        L = 0  # 车辆轴距

        # 更新速度和朝向
        self.v += a * dt
        self.heading_theta += (self.v / L) * np.tan(np.radians(delta)) * dt

        # 更新位置
        self.x += self.v * np.cos(self.heading_theta) * dt
        self.y += self.v * np.sin(self.heading_theta) * dt

        return {
            "x": self.x,
            "y": self.y,
            "heading_theta": self.heading_theta,
        }

    def reset(self, x, y, v, theta, t):
        self.x = x
        self.y = y
        self.v = v
        self.heading_theta = theta