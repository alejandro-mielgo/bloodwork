import logging
from logistic import execute_lr



#__ Config ___________________________________________________________________________________
OPTIMIZE_MODEL : bool = False
TRAIN_MODEL : bool = False
SAVE_MODEL : bool = False
SAVE_PREDICTIONS : bool = True

TRAIN_TEST_SPLIT : float = 0.85
SEED_NUMBER : int = 0


logging.basicConfig(
    filename='bloodwork.log',
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

if __name__ == "__main__":
 pass
