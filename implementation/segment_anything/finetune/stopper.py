class Stopper:
    def __init__(self, cfg):
        self.cfg = cfg
        pass

    def record_metrics(self, metrics, split):
        raise NotImplementedError()

    def should_stop(self):
        raise NotImplementedError()


class DummyStopper(Stopper):

    def record_metrics(self, metrics, split):
        pass

    def should_stop(self):
        return False


class EarlyStopper(Stopper):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.patience = cfg.schedule.stopper.patience
        self.mode = cfg.schedule.stopper.mode
        self.metric = cfg.schedule.stopper.metric
        self.split = cfg.schedule.stopper.split
        self.best_metric = None
        self.counter = 0

    def record_metrics(self, metrics, split):
        if split != self.split:
            return  # only monitor metrics for the split we are monitoring
        if self.best_metric is None:
            self.best_metric = metrics[self.metric]
        elif (self.mode == 'min' and metrics[self.metric] < self.best_metric)\
                or (self.mode == 'max' and metrics[self.metric] > self.best_metric):
            self.best_metric = metrics[self.metric]
            self.counter = 0
        else:
            self.counter += 1

    def should_stop(self):
        return self.counter >= self.patience
