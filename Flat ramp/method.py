import numpy as np
import traci
import math
import random
class SimulationControl:
    def __init__(self, config_path):
        self.config_path = config_path
        self.gap = float('inf')  # 初始化间隙为无限大，表示开始时没有任何间隙信息
        self.gap_rear = float('inf')  # 初始化间隙为无限大，表示开始时没有任何间隙信息
        self.initial_velocity = None
        self.start_time = None  # 初次检测到ego接近匝道出口（距出口十米处）的时间
        self.start_time_recorded = False  # 用于检测是否ego已经有了self.start_time的记录，有了则保持该值不变
        self.start_merge = None  # 初次检测到ego合流成功时的时间
        self.is_success = False  # 记录ego合流是否成功
        self.velocity = None
        self.near_veh = None
        self.near_veh_rear = None
        self.observation = np.empty(8)
        self.last_teleport_number = 0

    def check_collision(self):
        current_teleport_number = traci.simulation.getCollidingVehiclesNumber()
        if current_teleport_number > self.last_teleport_number:
            self.last_teleport_number = current_teleport_number
            print(f"发生{self.last_teleport_number}次碰撞")
            return True
        return False

    def start(self, gui=True):
        # 设置 SUMO 二进制路径，选择是否使用 GUI
        sumoBinary = 'D:/Sumo/bin/sumo'  # 默认不使用 GUI
        if gui:
            sumoBinary = 'D:/Sumo/bin/sumo-gui'  # 使用 GUI

        sumoCmd = [
            sumoBinary,
            '-c', self.config_path,
            '--step-length', '0.1',
            '--quit-on-end', 'true',  # 确保 SUMO 在仿真结束时自动退出
            '--no-step-log', 'true',
            '--collision.action', 'teleport',  # 启用碰撞检测并传送碰撞车辆
            '--collision.check-junctions', 'true',  # 在路口也进行碰撞检测
            '--start', 'true',
            '--default.action-step-length', '0.1'  # 防止弹出日志相关的步骤窗口
        ]

        traci.start(sumoCmd)  # 使用 TraCI 启动 SUMO

    def reset(self):
        traci.load(['-c', self.config_path, '--step-length', '0.1', '--quit-on-end', 'true', '--no-step-log', 'true',
                    '--start', 'true', '--collision.action', 'teleport', '--collision.check-junctions', 'true',
                    '--default.action-step-length', '0.1'])
        traci.simulationStep()

    def add_ego(self):
        # 添加车辆到仿真
        depart_time = random.randint(25, 75)  # 生成25到75之间的随机数
        traci.vehicle.add(vehID="ego", typeID="Ego", routeID="ramp-lane2", depart=depart_time)
        traci.vehicle.setColor("ego",  (255, 255, 255))

    def update_gap(self):
        closest_distance = float('inf')
        current_speed = traci.vehicle.getSpeed("ego")
        desired_gap = max(20, 2 * current_speed)  # 基于当前速度动态计算所需的安全间隙

        for veh_id in traci.vehicle.getIDList():
            if veh_id != "ego" and traci.vehicle.getLaneID(veh_id) == traci.vehicle.getLaneID("ego"):
                distance = traci.vehicle.getLanePosition(veh_id) - traci.vehicle.getLanePosition("ego")
                if distance > 0 and distance < closest_distance:  # 仅考虑前方车辆
                    closest_distance = distance
                    self.near_veh = veh_id
            # 注意：无需在 else 分支内再次判断 distance

        if closest_distance == float('inf'):  # 如果没有找到任何前车
            closest_distance = desired_gap
            self.near_veh = "ego"

        self.gap = closest_distance

    def update_gap_rear(self):
        closest_distance = float('inf')
        current_speed = traci.vehicle.getSpeed("ego")
        desired_gap = max(20, 2 * current_speed)

        for veh_id in traci.vehicle.getIDList():
            if veh_id != "ego" and traci.vehicle.getLaneID(veh_id) == traci.vehicle.getLaneID("ego"):
                distance = traci.vehicle.getLanePosition("ego") - traci.vehicle.getLanePosition(veh_id)
                if distance > 0 and distance < closest_distance:  # 仅考虑后方车辆
                    closest_distance = distance
                    self.near_veh_rear = veh_id
            elif veh_id != "ego" and traci.vehicle.getLaneID(veh_id) == "lane0_0":
                veh_position = traci.vehicle.getPosition(veh_id)  # 获取其他车辆位置
                ego_position = traci.vehicle.getPosition("ego")
                distance = self.calculate_distance(ego_position[0], ego_position[1], veh_position[0], veh_position[1])
                if distance < closest_distance:
                    closest_distance = distance
                    self.near_veh_rear = veh_id

        if closest_distance == float('inf'):  # 如果没有找到任何后车
            closest_distance = desired_gap
            self.near_veh_rear = "ego"

        self.gap_rear = closest_distance

    def calculate_distance(self, x1, y1, x2, y2):
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


    def merge_time(self):  # 该方法用于返回ego匝道合流的总时间
        self.merge_time_1()
        self.merge_time_2()
        current_road = traci.vehicle.getRoadID('ego')
        if current_road == "lane2":
            merge_time = self.start_merge - self.start_time
        else:
            merge_time = traci.simulation.getTime() - self.start_time
        return merge_time

    def merge_time_1(self):  # 该方法用于记录ego接近匝道出口的时间，并在之后的重复调用中保持该值self.start_time不变
        if not self.start_time_recorded:
            self.start_time = traci.simulation.getTime()
            self.initial_velocity = traci.vehicle.getSpeed("ego")
            self.start_time_recorded = True
        self.velocity = traci.vehicle.getSpeed("ego")


    def merge_time_2(self):# 该方法用于记录ego在调用该方法时初次检测到变道至lane2的时间，并在之后的重复调用中保持该值self.start_merge不变
        if not self.is_success and self.start_merge is None:
            str1 = str(traci.vehicle.getRoadID("ego"))
            if str1.startswith("lane2"):
                self.start_merge = traci.simulation.getTime()
                self.is_success = True

    def merge_time_clear(self):#该方法用于每次仿真后清除已有的合流时间信息
        self.start_time = None
        self.start_time_recorded = False
        self.start_merge = None
        self.is_success = False
    def get_state(self):
        # 假设 state 包括位置、速度和加速度
        self.update_gap()  # 调用 Ego 类的 update_gap 方法

        self.observation[0] = traci.vehicle.getPosition("ego")[0]  # 更新并记录ego的x坐标
        self.observation[1] = traci.vehicle.getPosition("ego")[1]  # 更新并记录ego的y坐标
        self.observation[2] = traci.vehicle.getSpeed("ego")  # 更新并记录ego的速度
        self.observation[3] = traci.vehicle.getAcceleration("ego")  # 更新并记录ego的加速度
        current_road = traci.vehicle.getRoadID('ego')
        if current_road == "lane2":  # 假设有一个方法来检测间隙信息的有效性
            self.observation[4] = self.gap  # 更新并记录当前时刻与ego最近的车（near_veh）之间的距离
            self.observation[5] = self.gap_rear  # 更新并记录当前时刻与ego最近的车（near_veh）之间的距离
            self.observation[6] = traci.vehicle.getSpeed(self.near_veh)  # 更新并记录当前时刻与ego最近的前车（near_veh）的速度
            self.observation[7] = traci.vehicle.getSpeed(self.near_veh_rear)  # 更新并记录当前时刻与ego最近的后车（near_veh_rear）的速度
        else:
            # ego合流成功前设为0
            self.observation[4] = 0
            self.observation[5] = 0
            self.observation[6] = 0
            self.observation[7] = 0
        return self.observation

    def calculate_reward(self):
        merge_time = self.merge_time()
        current_speed = traci.vehicle.getSpeed("ego")
        acceleration = traci.vehicle.getAcceleration("ego")

        # 效率奖励：基于归一化的速度差和merge time
        speed_diff = (current_speed - self.initial_velocity) / max(self.initial_velocity, 1)
        efficiency_reward = -0.5 * merge_time + 0.5 * speed_diff

        # 舒适性奖励：基于加速度的平方，鼓励平稳驾驶
        comfort_reward = -0.1 * (acceleration ** 2)

        # 安全奖励：基于动态调整的间隙
        if self.check_collision():
            safety_reward = -99999999
        # 安全奖励：基于动态调整的间隙
        else:
            self.update_gap()
            self.update_gap_rear()
            desired_gap = max(20, 2 * current_speed)  # 基于当前速度动态调整期望间隙
            safety_reward = max(0, self.gap - desired_gap) + max(0, self.gap_rear - desired_gap)

        # 合流成功奖励
        merge_success_reward = 50 if self.is_success else -50  # 成功合流奖励，失败则惩罚

        total_reward = efficiency_reward + comfort_reward + safety_reward + merge_success_reward
        return total_reward


