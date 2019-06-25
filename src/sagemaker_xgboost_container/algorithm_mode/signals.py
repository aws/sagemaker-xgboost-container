import ctypes
import logging
import os
import sys


def _load_lib(lib_prefix):
    """
    Loads the .so lib built for the C++ code in signals.cpp
    Args:
        lib_prefix (str): .so file name that needs to be looked up, expects the prefix without .so part
    Returns:
        ctypes.CDLL: the CDLL that can be used to invoke the C++ function in python modules
    """
    LIB_DIR = os.path.dirname(os.path.abspath(__file__))
    major_version, minor_version = sys.version_info[0], sys.version_info[1]
    if major_version < 3:
        # Library name and path at buld time for Python2.7
        LIB_NAME = '{}.so'.format(lib_prefix)
    else:
        # Library name and path at build time for Python3
        # Documentation: http://legacy.python.org/dev/peps/pep-3149/
        LIB_NAME = '{}.cpython-{}{}m.so'.format(lib_prefix, major_version, minor_version)

    # At runtime the library will be in the same directory as signals.py, so this
    # is the first place we try

    lib_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), LIB_NAME)

    logging.info("Looking for signals.cpp file in {}".format(lib_path))

    try:
        return ctypes.CDLL(lib_path)
    except:
        lib_path = [os.path.join(p, LIB_NAME) for p in LIB_DIR]
        lib_path = [p for p in lib_path if os.path.exists(p) and os.path.isfile(p)]
        logging.info("Now trying: {}".formaat(lib_path[0]))
        return ctypes.CDLL(lib_path[0])


def install_terminate_and_signal_handlers():
    __LIB = _load_lib('signals_lib')
    __LIB.install_terminate_and_signal_handlers()
