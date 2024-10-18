from time import time


class LabelAnnealer:
    def __init__(self, agent_logger, final_timesteps, final_labels, pretrain_labels) -> None:
        self._agent_logger = agent_logger
        self._final_timesteps = final_timesteps
        self._final_labels = final_labels
        self._pretrain_labels = pretrain_labels

    @property
    def n_desired_labels(self):
        # Decay from 1 to 0
        exp_decay_frac = 0.01 ** (self._agent_logger._timesteps_elapsed /
                                  self._final_timesteps)
        pretrain_frac = self._pretrain_labels / self._final_labels
        # Start with 0.25 and anneal to 0.99
        desired_frac = pretrain_frac + \
            (1 - pretrain_frac) * (1 - exp_decay_frac)
        return desired_frac * self._final_labels


class ConstantLabelSchedule:
    def __init__(self, pretrain_labels, seconds_between_labels=3.0) -> None:
        self._started_at = None  # Don't initialize until we call n_desired_labels
        self._seconds_between_labels = seconds_between_labels
        self._pretrain_labels = pretrain_labels

    @property
    def n_desired_labels(self):
        if self._started_at is None:
            self._started_at = time()
        return self._pretrain_labels + (time() - self._started_at) / self._seconds_between_labels
