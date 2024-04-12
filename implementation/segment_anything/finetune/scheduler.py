import torch


class Scheduler:
    def __init__(self, cfg, optimizer):
        self.cfg = cfg
        self.optimizer = optimizer

    def observe_metrics(self, metrics, split):
        raise NotImplementedError()

class DummyScheduler(Scheduler):
    def observe_metrics(self, metrics, split):
        pass

class ReduceLROnPlateauScheduler(Scheduler):
    def __init__(self, cfg, optimizer):
        super().__init__(cfg, optimizer)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode=cfg.schedule.scheduler.mode,
            factor=cfg.schedule.scheduler.factor,
            patience=cfg.schedule.scheduler.patience,
            threshold=cfg.schedule.scheduler.threshold,
        )
        self.split = cfg.schedule.scheduler.split
        self.metric = cfg.schedule.scheduler.metric

    def observe_metrics(self, metrics, split):
        if split != self.split:
            return
        self.scheduler.step(metrics[self.metric])

