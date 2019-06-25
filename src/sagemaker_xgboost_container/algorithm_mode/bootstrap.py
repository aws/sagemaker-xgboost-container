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
import logging
import os
import psutil
from shutil import copyfile
import subprocess


from sagemaker_xgboost_container.constants.xgb_constants import FNULL


HADOOP_PREFIX = os.environ['HADOOP_PREFIX']


def file_prepare():
    # src = '/tmp/hdfs-site.xml'
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hdfs-site.xml")
    dst = HADOOP_PREFIX + '/etc/hadoop/hdfs-site.xml'
    copyfile(src, dst)

    # src= '/tmp/core-site.xml'
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "core-site.xml")
    dst = HADOOP_PREFIX + '/etc/hadoop/core-site.xml'
    copyfile(src, dst)

    # src= '/tmp/yarn-site.xml'
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yarn-site.xml")
    dst = HADOOP_PREFIX + '/etc/hadoop/yarn-site.xml'
    copyfile(src, dst)


def cluster_config(num_hosts, current_host, master_host, master_ip):
    hdfs_site_file_path = HADOOP_PREFIX + "/etc/hadoop/hdfs-site.xml"
    core_site_file_path = HADOOP_PREFIX + "/etc/hadoop/core-site.xml"
    yarn_site_file_path = HADOOP_PREFIX + "/etc/hadoop/yarn-site.xml"

    # configure ip address for name node
    with open(core_site_file_path, 'r') as core_file:
        file_data = core_file.read()
    file_data = file_data.replace('nn_uri', master_ip)
    with open(core_site_file_path, 'w') as core_file:
        core_file.write(file_data)

    # configure hostname for RM and NM
    with open(yarn_site_file_path, 'r') as yarn_file:
        file_data = yarn_file.read()
    file_data = file_data.replace('rm_hostname', master_host)
    file_data = file_data.replace('nm_hostname', current_host)
    with open(yarn_site_file_path, 'w') as yarn_file:
        yarn_file.write(file_data)

    # configure yarn resource limitation
    mem = int(psutil.virtual_memory().total/(1024*1024))        # total physical memory in mb
    cores = psutil.cpu_count(logical=False)                     # total physical cores

    minimum_allocation_mb = '1'
    maximum_allocation_mb = str(mem)
    minimum_allocation_vcores = '1'
    maximum_allocation_vcores = str(cores)
    # add some residual in memory due to rounding in memory allocation
    memory_mb_total = str(mem+2048)
    # virtualized 32x cores to ensure core allocations
    cpu_vcores_total = str(cores*32)

    with open(yarn_site_file_path, 'r') as yarn_file:
        file_data = yarn_file.read()
    file_data = file_data.replace('minimum_allocation_mb', minimum_allocation_mb)
    file_data = file_data.replace('maximum_allocation_mb', maximum_allocation_mb)
    file_data = file_data.replace('minimum_allocation_vcores', minimum_allocation_vcores)
    file_data = file_data.replace('maximum_allocation_vcores', maximum_allocation_vcores)
    file_data = file_data.replace('memory_mb_total', memory_mb_total)
    file_data = file_data.replace('cpu_vcores_total', cpu_vcores_total)
    with open(yarn_site_file_path, 'w') as yarn_file:
        yarn_file.write(file_data)

    logging.info("Finished Yarn configuration files setup.\n")


def start_daemons(master_host, current_host):
    logging.info("Current host: {}".format(current_host))
    logging.info("Master host: {}".format(master_host))

    cmd_namenode_format = HADOOP_PREFIX + '/bin/hdfs namenode -format'
    cmd_start_namenode = HADOOP_PREFIX + '/sbin/hadoop-daemon.sh start namenode'
    cmd_start_resourcemanager = HADOOP_PREFIX + '/sbin/yarn-daemon.sh start resourcemanager'
    cmd_start_datanode = HADOOP_PREFIX + '/sbin/hadoop-daemon.sh start datanode'
    cmd_start_nodemanager = HADOOP_PREFIX + '/sbin/yarn-daemon.sh start nodemanager'

    if current_host == master_host:
        subprocess.call(cmd_namenode_format, shell=True, stdout=FNULL, stderr=FNULL)
        subprocess.call(cmd_start_namenode, shell=True, stderr=FNULL)
        subprocess.call(cmd_start_resourcemanager, shell=True, stderr=FNULL)
        subprocess.call(cmd_start_datanode, shell=True, stderr=FNULL)
        subprocess.call(cmd_start_nodemanager, shell=True, stderr=FNULL)
    else:
        subprocess.call(cmd_start_datanode, shell=True, stderr=FNULL)
        subprocess.call(cmd_start_nodemanager, shell=True, stderr=FNULL)


if __name__ == "__main__":
    pass
