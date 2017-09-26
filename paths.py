import os


def check(dirname):
    """This function creates a directory
    in case it doesn't exist"""
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return dirname


# The project directory
EVAL_DIR = os.path.dirname(os.path.realpath(__file__))

# Folder with datasets
DATASETS_ROOT = check(os.path.join(EVAL_DIR, 'Datasets'))

# Where the checkpoints are stored
CKPT_ROOT = check(os.path.join(EVAL_DIR, 'archive/'))

# Where the logs are stored
LOGS = check(os.path.join(EVAL_DIR, 'Logs/Experiments/'))

# Where the algorithms visualizations are dumped
RESULTS_DIR = check(os.path.join(EVAL_DIR, 'Results/'))

# Where the imagenet weights are located
INIT_WEIGHTS_DIR = check(os.path.join(EVAL_DIR, 'Weights_imagenet/'))

# Where the demo images are located
DEMO_DIR = check(os.path.join(EVAL_DIR, 'Demo/'))
