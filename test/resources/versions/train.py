import sys

import pkg_resources

PYTHON_MAJOR_VERSION = 3
PYTHON_MINOR_VERSION = 12
REQUIREMENTS = """\
Flask==3.1.3
Pillow==12.2.0
PyYAML==6.0.1
boto3==1.38.0
botocore==1.38.0
cryptography==46.0.7
gunicorn==25.3.0
matplotlib==3.10.9
multi-model-server==1.1.2
numpy==2.1.0
pandas==2.2.3
psutil==7.0.0
pyarrow==22.0.0
python-dateutil==2.9.0
retrying==1.3.3
sagemaker-containers==2.8.6.post2
sagemaker-inference==1.5.5
scipy==1.15.0
scikit-learn==1.8.0
urllib3==2.7.0
Werkzeug==3.1.8
certifi==2025.4.26
gevent==26.4.0
""".strip()


def assert_python_version(major, minor):
    assert sys.version_info.major == major and sys.version_info.minor == minor


def assert_package_version(package_name, version):
    installed_version = pkg_resources.get_distribution(package_name).version
    error_message = (
        f"{package_name} requires {version} but {installed_version} is installed."
    )
    assert version == installed_version, error_message


def parse_requirements(requirements):
    for package_equals_version in requirements.split("\n"):
        package, version = package_equals_version.split("==")
        yield package, version


if __name__ == "__main__":
    assert_python_version(PYTHON_MAJOR_VERSION, PYTHON_MINOR_VERSION)
    for package, version in parse_requirements(REQUIREMENTS):
        assert_package_version(package, version)
