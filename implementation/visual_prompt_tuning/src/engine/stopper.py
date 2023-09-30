from ..utils import logging

logger = logging.get_logger('visual_prompt')


def get_stopper(cfg):
    if cfg.SOLVER.TOTAL_EPOCH is not None:
        if cfg.SOLVER.PATIENCE < cfg.SOLVER.TOTAL_EPOCH:
            return EarlyEpochStopper(cfg)
        # early stopping is irrelevant when patience is >= epoch max
        return EpochStopper(cfg)
    if cfg.SOLVER.PATIENCE is not None:
        return EarlyStopper(cfg)
    raise ValueError("No stopping criterion specified.")

class EarlyStopper:
    """Indicates whether we should stop training early based on a stopping criterion and a patience."""
    def __int__(self, cfg):
        # todo: implement stopping based on validation loss
        self.cfg = cfg  # general config
        self.criterion = cfg.SOLVER.CRITERION  # metric to use for early stopping
        self.patience = cfg.SOLVER.PATIENCE  # number of epochs to wait for improvement
        self.best_score = None
        self.counter = 0

    def __call__(self, metrics: dict) -> bool:
        # metrics expects at least keys that match self.criterion and "epoch"
        score = metrics[self.criterion]
        if self.criterion == "loss":  # or any other metric where less is better
            score = -score  # invert the score because that simplifies comparisons
        if self.best_score is None:
            self.best_score = score
            return False
        if score > self.best_score:
            logger.info(f'Best epoch {metrics["epoch"] + 1}: best metric: {score:.3f}')
            self.best_score = score
            return False
        self.counter += 1
        if self.counter >= self.patience:
            if self.cfg.DBG:
                # todo: is it necessary to improve this log?
                logger.info(f"Stopping early.")
            return True
        return False


class EpochStopper:
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, metrics: dict) -> bool:
        if metrics["epoch"] < self.cfg.SOLVER.TOTAL_EPOCH:
            return False
        if self.cfg.DBG:
            logger.info(f"Stopping at epoch limit.")
        return True

class EarlyEpochStopper(EarlyStopper):
    def __init__(self, cfg):
        super().__init__(cfg)

    def __call__(self, metrics: dict) -> bool:
        if metrics["epoch"] < self.cfg.SOLVER.TOTAL_EPOCH:
            return super().__call__(metrics)  # only consider early stopping if we are not at the epoch limit
        if self.cfg.DBG:
            logger.info(f"Stopping at epoch limit.")
        return True






