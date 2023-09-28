from ..utils import logging

logger = logging.get_logger('visual_prompt')

class EarlyStopper:
    """Indicates whether we should stop training early based on a stopping criterion and a patience."""
    def __int__(self, cfg, criterion: str = "loss", patience: int = 7, verbose: bool = False):
        # todo: implement stopping based on validation loss
        self.cfg = cfg  # general config
        self.criterion = criterion
        self.patience = patience
        self.verbose = verbose
        self.best_score = None
        self.counter = 0

    def __call__(self, metrics: dict) -> bool:
        # metrics expects at least a key that matches f"{self.criterion}"
        score = metrics[self.criterion]
        if self.criterion == "loss":  # or any other metric where less is better
            score = -score  # invert the score because that simplifies comparisons
        if self.best_score is None:
            self.best_score = score
            return False
        if score > self.best_score:
            self.best_score = score
            return False
        self.counter += 1
        if self.counter >= self.patience:
            if self.verbose:
                # todo: is it necessary to improve this log?
                logger.info(f"Stopping early.")
            return True
        return False

class EpochStopper:
    def __init__(self, cfg, verbose: bool = False):
        self.cfg = cfg
        self.verbose = verbose

    def __call__(self, metrics: dict) -> bool:
        raise NotImplementedError()








