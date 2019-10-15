import shlex
import subprocess
import sys
import os.path

from sagemaker_containers.beta.framework import trainer
from sagemaker_xgboost_container import serving

if sys.argv[1] == 'serve':
    serving.main()
    subprocess.call(['tail', '-f', '/dev/null'])
elif sys.argv[1] == 'train':
    trainer.train()
