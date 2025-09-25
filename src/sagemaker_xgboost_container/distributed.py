# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License'). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the 'license' file accompanying this file. This file is
# distributed on an 'AS IS' BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""
Contains most of the wrapping code for XGBoost distributed training and Rabit support.
This is heavily inspired by the Dask version of XGBoost.
Some of this code should be made simpler once the XGBoost library is improved.
"""
# Dask-based replacement for distributed.py
import logging
import sys
from dask.distributed import Client, as_completed
from dask import delayed
import xgboost as xgb

LOCAL_HOSTNAME = "127.0.0.1"

def wait_hostname_resolution(sm_hosts):
    """Wait for hostname resolution - simplified for Dask"""
    pass  # Dask handles this internally

def rabit_run(exec_fun, args, include_in_training, hosts, current_host, 
              first_port=None, second_port=None, max_connect_attempts=None,
              connect_retry_timeout=3, update_rabit_args=False):
    """Run execution using Dask instead of rabit"""
    
    if not include_in_training:
        logging.warning("Host {} not being used for distributed training.".format(current_host))
        sys.exit(0)
    
    # Use Dask client for coordination
    scheduler_address = f"{hosts[0]}:{first_port or 8786}"
    
    with Client(scheduler_address) as client:
        if update_rabit_args:
            args.update({"is_master": client.scheduler_info()["address"] == scheduler_address})
        
        # Execute function in distributed manner
        future = client.submit(exec_fun, **args)
        return future.result()

class DaskHelper(object):
    def __init__(self, client, current_host):
        self.client = client
        self.current_host = current_host
        self.is_master = True  # Simplified
        
    def synchronize(self, data):
        """Synchronize data using Dask"""
        futures = self.client.scatter([data] * len(self.client.scheduler_info()["workers"]))
        return self.client.gather(futures)

class Rabit(object):
    def __init__(self, hosts, current_host=None, master_host=None, port=None, 
                 max_connect_attempts=None, connect_retry_timeout=3):
        self.hosts = hosts
        self.current_host = current_host or LOCAL_HOSTNAME
        self.master_host = master_host or hosts[0]
        self.port = port or 8786
        self.client = None
        
    def start(self):
        scheduler_address = f"{self.master_host}:{self.port}"
        self.client = Client(scheduler_address)
        return DaskHelper(self.client, self.current_host)
        
    def stop(self):
        if self.client:
            self.client.close()
            
    def __enter__(self):
        return self.start()
        
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stop()