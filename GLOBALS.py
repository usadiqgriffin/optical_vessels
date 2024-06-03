import os
import pygit2
import getpass
GIT_BRANCH_NAME = pygit2.Repository(".").head.shorthand
linux_user = getpass.getuser()


EXPERIMENT_OUTPUT_DIR = os.path.join(
    f"/teamspace/studios/this_studio/dl_python/cv/optical_vessels/{linux_user}",
    "optical_vessels",
    GIT_BRANCH_NAME
)

CHECKPOINT_DIR = os.path.join(
    EXPERIMENT_OUTPUT_DIR,
    "checkpoint"
)
CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "checkpoint"
)