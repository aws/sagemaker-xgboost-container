import pkg_resources
import sys


PYTHON_MAJOR_VERSION = 3
PYTHON_MINOR_VERSION = 7
REQUIREMENTS = """\
Flask==1.1.1
Pillow==9.1.1
PyYAML==5.4.1
boto3==1.17.52
botocore==1.20.52
conda==4.10.1
cryptography==35.0.0
gunicorn==19.10.0
matplotlib==3.4.1
multi-model-server==1.1.2
numpy==1.21.6
pandas==1.2.4
psutil==5.6.7
pyarrow==1.0.1
python-dateutil==2.8.1
requests==2.25.1
retrying==1.3.3
sagemaker-containers==2.8.6.post2
sagemaker-inference==1.5.5
scikit-learn==0.24.1
scipy==1.6.2
smdebug==1.0.29
urllib3==1.26.5
wheel==0.36.2
jinja2==2.10.2
MarkupSafe==1.1.1
Werkzeug==0.15.6
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
