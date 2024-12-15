import time

Secs = float


class StepTimer:
    def __init__(self, step: Secs, start_time: Secs = 0):
        self.step = step
        self.time = start_time
        self.unresettable_time = start_time

    def step_forward(self):
        self.time += self.step
        self.unresettable_time += self.step

    def reset(self, start_time: Secs = 0):
        self.time = start_time
