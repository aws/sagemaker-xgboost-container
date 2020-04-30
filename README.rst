===========================
SageMaker XGBoost Container
===========================

SageMaker XGBoost Container is an open source library for making the
XGBoost framework run on Amazon SageMaker.

This repository also contains Dockerfiles which install this library and dependencies
for building SageMaker XGBoost Framework images.

The SageMaker team uses this repository to build its official XGBoost Framework image. To use this image on SageMaker,
see `Python SDK <https://github.com/aws/sagemaker-python-sdk>`__.
For end users, this repository is typically of interest if you need implementation details for
the official image, or if you want to use it to build your own customized XGBoost Framework image.

Table of Contents
-----------------

#. `Getting Started <#getting-started>`__
#. `Building your Image <#building-your-image>`__
#. `Running the tests <#running-the-tests>`__

Getting Started
---------------

Prerequisites
~~~~~~~~~~~~~

Make sure you have installed all of the following prerequisites on your
development machine:

- `Docker <https://www.docker.com/>`__

Recommended
^^^^^^^^^^^

-  A Python environment management tool (e.g.
   `PyEnv <https://github.com/pyenv/pyenv>`__,
   `VirtualEnv <https://virtualenv.pypa.io/en/stable/>`__)

Building your image
-------------------

`Amazon SageMaker <https://aws.amazon.com/documentation/sagemaker/>`__
utilizes Docker containers to run all training jobs & inference endpoints.

The Docker images are built from the Dockerfiles specified in
`Docker/ <https://github.com/aws/sagemaker-xgboost-container/tree/master/docker>`__.

The Docker files are grouped based on XGboost version and separated
based on Python version and processor type.

The Docker images, used to run training & inference jobs, are built from
both corresponding "base" and "final" Dockerfiles.

Base Images
~~~~~~~~~~~

The "base" Dockerfile encompass the installation of the framework and all of the dependencies
needed.

Tagging scheme is based on <SageMaker-XGBoost-version>-cpu-py3 (e.g. |XGBoostLatestVersion|-cpu-py3), where
 <SageMaker-XGBoost-version> is comprised of <XGBoost-version>-<SageMaker-version>.

All "final" Dockerfiles build images using base images that use the tagging scheme
above.

If you want to build your base docker image, then use:

::

    # All build instructions assume you're building from the root directory of the sagemaker-xgboost-container.

    # CPU
    docker build -t xgboost-container-base:<SageMaker-XGBoost-version>-cpu-py3 -f docker/<SageMaker-XGBoost-version>/base/Dockerfile.cpu .

.. parsed-literal::

    # Example

    # CPU
    docker build -t xgboost-container-base:|XGBoostLatestVersion|-cpu-py3 -f docker/|XGBoostLatestVersion|/base/Dockerfile.cpu .


Final Images
~~~~~~~~~~~~

The "final" Dockerfiles encompass the installation of the SageMaker specific support code.

All "final" Dockerfiles use base images for building.

These "base" images are specified with the naming convention of
xgboost-container-base:<SageMaker-XGBoost-version>-cpu-py3.

Before building "final" images:

Build your "base" image. Make sure it is named and tagged in accordance with your "final"
Dockerfile.

::

    # Create the SageMaker XGBoost Container Python package.
    cd sagemaker-xgboost-container
    python setup.py bdist_wheel --universal

If you want to build "final" Docker images, then use:

::

    # All build instructions assume you're building from the root directory of the sagemaker-xgboost-container.

    # CPU
    docker build -t <image_name>:<tag> -f docker/<xgboost-version>/final/Dockerfile.cpu .

.. parsed-literal::

    # Example

    # CPU
    docker build -t preprod-xgboost-container:|XGBoostLatestVersion|-cpu-py3 -f docker/|XGBoostLatestVersion|/final/Dockerfile.cpu .

Running the tests
-----------------

Running the tests requires installation of the SageMaker XGBoost Framework container code and its test
dependencies.

::

    git clone https://github.com/aws/sagemaker-xgboost-container.git
    cd sagemaker-xgboost-container
    pip install -e .[test]

Conda is also required and can be installed by following the instructions at https://conda.io/projects/conda/en/latest/user-guide/install/index.html. For convenience, the Linux installation commands are provided as an example.

::

    curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -bfp /miniconda3
    rm Miniconda3-latest-Linux-x86_64.sh
    export PATH=/miniconda3/bin:${PATH}
    conda update -y conda

Tests are defined in
`test/ <https://github.com/aws/sagemaker-xgboost-container/tree/master/test>`__
and include unit, local integration, and SageMaker integration tests.

Unit Tests
~~~~~~~~~~

If you want to run unit tests, then use:

::

    # All test instructions should be run from the top level directory

    pytest test/unit

    # or you can use tox to run unit tests as well as flake8 and code coverage

    tox
    tox -e py3-xgboost1.0,flake8
    tox -e py3-xgboost0.90,py3-xgboostlatest


Local Integration Tests
~~~~~~~~~~~~~~~~~~~~~~~

Running local integration tests require `Docker <https://www.docker.com/>`__ and `AWS
credentials <https://docs.aws.amazon.com/sdk-for-java/v1/developer-guide/setup-credentials.html>`__,
as the local integration tests make calls to a couple AWS services. The local integration tests and
SageMaker integration tests require configurations specified within their respective
`conftest.py <https://github.com/aws/sagemaker-xgboost-container/blob/master/test/conftest.py>`__.

Before running local integration tests:

#. Build your Docker image.
#. Pass in the correct pytest arguments to run tests against your Docker image.

If you want to run local integration tests, then use:

::

    # Required arguments for integration tests are found in test/conftest.py

    pytest test/integration/local --docker-base-name <your_docker_image> \
                      --tag <your_docker_image_tag> \
                      --py-version <2_or_3> \
                      --framework-version <xgboost-version>

.. parsed-literal::

    # Example
    pytest test/integration/local --docker-base-name preprod-xgboost-container ``\``
                      --tag |XGBoostLatestVersion|-cpu-py3 ``\``
                      --py-version 3 ``\``
                      --framework-version |XGBoostLatestVersion|

SageMaker Integration Tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~

SageMaker integration tests require your Docker image to be within an `Amazon ECR repository <https://docs
.aws.amazon.com/AmazonECS/latest/developerguide/ECS_Console_Repositories.html>`__.

The Docker base name is your `ECR repository namespace <https://docs.aws.amazon
.com/AmazonECR/latest/userguide/Repositories.html>`__.

The instance type is your specified `Amazon SageMaker Instance Type
<https://aws.amazon.com/sagemaker/pricing/instance-types/>`__ that the SageMaker integration test will run on.

Before running SageMaker integration tests:

#. Build your Docker image.
#. Push the image to your ECR repository.
#. Pass in the correct pytest arguments to run tests on SageMaker against the image within your ECR repository.

If you want to run a SageMaker integration end to end test on `Amazon
SageMaker <https://aws.amazon.com/sagemaker/>`__, then use:

::

    # Required arguments for integration tests are found in test/conftest.py

    pytest test/integration/sagemaker --aws-id <your_aws_id> \
                           --docker-base-name <your_docker_image> \
                           --instance-type <amazon_sagemaker_instance_type> \
                           --tag <your_docker_image_tag>

::

    # Example
    pytest test/integration/sagemaker --aws-id 12345678910 \
                           --docker-base-name preprod-xgboost-container \
                           --instance-type ml.m4.xlarge \
                           --tag 1.0

Contributing
------------

Please read
`CONTRIBUTING.md <https://github.com/aws/sagemaker-xgboost-container/blob/master/CONTRIBUTING.md>`__
for details on our code of conduct, and the process for submitting pull
requests to us.

License
-------

SageMaker XGboost Framework Container is licensed under the Apache 2.0 License. It is copyright 2019 Amazon
.com, Inc. or its affiliates. All Rights Reserved. The license is available at:
http://aws.amazon.com/apache2.0/

.. |XGBoostLatestVersion| replace:: 1.0-1