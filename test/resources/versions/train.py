import pkg_resources
import sys


PYTHON_MAJOR_VERSION = 3
PYTHON_MINOR_VERSION = 7
REQUIREMENTS = """\
Flask==1.1.1
Pillow==8.1.0
PyYAML==5.4
boto3==1.14.62
botocore==1.17.62
conda==4.9.0
cryptography==3.3.2
gunicorn==19.10.0
matplotlib==3.3.2
multi-model-server==1.1.1
numpy==1.19.2
pandas==1.1.3
psutil==5.6.7
pyarrow==0.16.0
python-dateutil==2.8.0
requests==2.25.1
retrying==1.3.3
sagemaker-containers==2.8.6.post2
sagemaker-inference==1.2.0
scikit-learn==0.23.2
scipy==1.5.3
smdebug==1.0.2
urllib3==1.25.9
wheel==0.35.1
""".strip()


def assert_python_version(major, minor):
    assert sys.version_info.major == major and sys.version_info.minor == minor


def assert_package_version(package_name, version):
    installed_version = pkg_resources.get_distribution(package_name).version
    error_message = f"{package_name} requires {version} but {installed_version} is installed."
    assert version == installed_version, error_message


def parse_requirements(requirements):
    for package_equals_version in requirements.split('\n'):
        package, version = package_equals_version.split("==")
        yield package, version


if __name__ == '__main__':
    assert_python_version(PYTHON_MAJOR_VERSION, PYTHON_MINOR_VERSION)
    for package, version in parse_requirements(REQUIREMENTS):
        assert_package_version(package, version)
