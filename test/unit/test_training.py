from __future__ import absolute_import

from mock import MagicMock, patch

from sagemaker_xgboost_container import training


def mock_training_env(current_host='algo-1', module_dir='s3://my/script', module_name='svm', **kwargs):
    return MagicMock(current_host=current_host, module_dir=module_dir, module_name=module_name, **kwargs)


@patch('sagemaker_containers.beta.framework.modules.run_module')
def test_single_machine(run_module):
    env = mock_training_env()
    training.train(env)

    run_module.assert_called_with('s3://my/script', env.to_cmd_args(), env.to_env_vars(), 'svm')
