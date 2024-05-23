import logging

import torch
import traci
import numpy as np

from Car import Ego
from method import SimulationControl
from ppo import PPO, Memory

# 设置日志
log_file = 'C:/Users/Mr.D/Desktop/Graduation thesis/Stereo ramp/Stereo ramp training_log.log'
config_path = 'C:/Users/Mr.D/Desktop/Graduation thesis/Stereo ramp/Stereo ramp.sumocfg'

with open(log_file, 'w') as file:  # 每次仿真前先清除上一次的日志文件
    file.truncate()
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')


def log_performance(epoch, reward, loss, merge_time, gap, gap_rear, avg_speed):
    logging.info(f"Epoch: {epoch}, Total Reward: {reward}, Loss: {loss}, Merge Time: {merge_time}, Gap: {gap}, Gap Rear: {gap_rear}, Avg Speed: {avg_speed}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 确保所有参数适配
    ppo = PPO(state_dim=8, lr=0.0001, betas=(0.9, 0.999), gamma=0.9, K_epochs=4, eps_clip=0.2, device=device,
              action_std=0.1)
    epoch = 1  # 初始化训练轮次
    sim_control = SimulationControl(config_path)
    sim_control.start()
    while True:  # 主循环，用于多次运行仿真
        memory = Memory()
        ego_vehicle = Ego('ego')
        total_reward = 0
        step_count = 0  # 计数器
        duration = 0
        sim_control.add_ego()
        gap = []
        gap_rear = []
        speed = []
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            step_count += 1  # 每个仿真步长增加计数器
            # 每10步执行一次动作选择和状态更新(1s)
            if 'ego' in traci.vehicle.getIDList():
                speed.append(traci.vehicle.getSpeed("ego"))
                if step_count % 10 == 0:
                    traci.vehicle.setSpeedMode("ego", 0)  # 关闭所有安全速度控制
                    sim_control.merge_time_1()
                    sim_control.merge_time_2()
                    sim_control.update_gap()
                    sim_control.update_gap_rear()

                    ego_vehicle.update_state()
                    state = sim_control.get_state()
                    state_tensor = torch.FloatTensor(state).to(device)
                    action = ppo.select_action(memory, state_tensor)
                    traci.vehicle.setSpeed('ego', action[0])

                    reward = sim_control.calculate_reward()
                    total_reward += reward
                    memory.rewards.append(reward)

                current_road = traci.vehicle.getRoadID('ego')
                if current_road == "lane2":
                    sim_control.update_gap()
                    sim_control.update_gap_rear()
                    for i in range(5):
                        gap.append(sim_control.gap)
                        gap_rear.append(sim_control.gap_rear)
                    if traci.vehicle.getLanePosition("ego") >= 50:
                        loss = ppo.update(memory)
                        merge_time = sim_control.merge_time()
                        sim_control.merge_time_clear()
                        avg_speed = sum(speed) / len(speed)
                        log_performance(epoch, total_reward, loss, merge_time, gap[0], gap_rear[0], avg_speed)
                        print(f"Epoch {epoch}: Total Reward = {total_reward}, Loss = {loss}, Merge Time = {merge_time}, gap = {gap[0]}, gap_rear = {gap_rear[0]}, avg_speed = {avg_speed}")
                        memory.clear_memory()
                        total_reward = 0

                        epoch += 1
                        sim_control.gap = float('inf')  # 初始化间隙为无限大，表示开始时没有任何间隙信息
                        sim_control.gap_rear = float('inf')  # 初始化间隙为无限大，表示开始时没有任何间隙信息
                        sim_control.initial_velocity = None
                        sim_control.desired_gap = 25  # 主道车辆的理想安全间隔
                        sim_control.desired_gap_rear = 25
                        sim_control.start_time = None  # 初次检测到ego接近匝道出口（距出口十米处）的时间
                        sim_control.start_time_recorded = False  # 用于检测是否ego已经有了self.start_time的记录，有了则保持该值不变
                        sim_control.start_merge = None  # 初次检测到ego合流成功时的时间
                        sim_control.is_success = False  # 记录ego合流是否成功
                        sim_control.velocity = None
                        sim_control.near_veh = None
                        sim_control.near_veh_rear = None
                        sim_control.observation = np.empty(8)  # 状态空间为8
                        sim_control.last_teleport_number = 0
                        break
            if sim_control.check_collision():  # 若发生碰撞，则立马开启下一轮仿真
                sim_control.last_teleport_number = 0
                sim_control.reset()
                break

        sim_control.reset()


if __name__ == '__main__':
    main()
