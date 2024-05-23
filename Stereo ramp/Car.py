
import traci

class Car:
    def __init__(self, vehID):
        self.vehID = vehID

    def update_state(self):
        self.position = traci.vehicle.getPosition(self.vehID)
        self.velocity = traci.vehicle.getSpeed(self.vehID)
        self.acceleration = traci.vehicle.getAcceleration(self.vehID)

class Ego(Car):
    def __init__(self, vehID):
        super().__init__(vehID)


