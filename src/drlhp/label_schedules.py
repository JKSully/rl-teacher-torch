from time import time
from abc import ABC, abstractmethod
# TODO: Redo these classes


class LabelAnnealer:
    def __init__(self, agent_logger, final_timesteps, final_labels, pretrain_labels) -> None:
        self._agent_logger = agent_logger
        self._final_timesteps = final_timesteps
        self._final_labels = final_labels
        self._pretrain_labels = pretrain_labels

    @property
    def n_desired_labels(self):
        exp_decay_frac = 0.01 ** (self._agent_logger._timesteps_elapsed /
                                  self._final_timesteps)
        pretrain_frac = self._pretrain_labels / self._final_labels
        # Start with 0.25 and anneal to 0.99
        desired_frac = pretrain_frac + \
            (1 - pretrain_frac) * (1 - exp_decay_frac)
        return desired_frac * self._final_labels


class LabelSchedule(ABC):
    def __init__(self, pretrain_labels: int) -> None:
        self._pretrain_labels = pretrain_labels
        self._stated_at = None
    
    @property 
    @abstractmethod
    def n_desired_labels(self):
        pass

    def start_timing(self):
        self._started_at = time()

    @property
    def time_elapsed(self):
        if self._started_at is None:
            return 0
        return time() - self._started_at
        

class ConstantLabelSchedule(LabelSchedule):
    def __init__(self, pretrain_labels: int, seconds_between_labels: float=3.0) -> None:
        super().__init__(pretrain_labels)
        self._seconds_between_labels = seconds_between_labels

    @property
    def n_desired_labels(self):
        return self._pretrain_labels + self.time_elapsed / self._seconds_between_labels
