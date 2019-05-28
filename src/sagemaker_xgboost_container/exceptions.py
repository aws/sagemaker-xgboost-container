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
import sys
import warnings

EXIT_CODE_SUCCESS = 0
EXIT_CODE_ALGORITHM_ERROR = 1
EXIT_CODE_CUSTOMER_ERROR = 2
EXIT_CODE_PLATFORM_ERROR = 3


# Note: This was copied from Algorithm internal SDK; this is a stub for ATC integration


def convert_to_algorithm_error(exception):
    """Converts the most recent exception to an AlgorithmError if not already
    a BaseSdkError.

    Returns:
         A BaseSdkError that represents the reason the algorithm failed.
    """
    if isinstance(exception, BaseSdkError):
        return exception
    elif "(Platform Error)" in str(exception):
        return PlatformError(
            "An unexpected error has occurred. Please try again. If the problem persists, contact AWS support.",
            caused_by=exception)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Suppress deprecation warning
        message = getattr(exception, 'message', str(exception))
    return AlgorithmError(message, exception)


def convert_to_customer_data_errors(exception, channel, content_type):
    """Convert exception from data iterators to customer errors.

    If exception is a BaseSdkError or not an input error, return the value of exception.

    :param exception: (Exception) exception instance
    :param channel: (str) data channel name
    :param content_type: (str) content type or None
    :return: (Exception) an instance of CustomerError or the value of exception parameter
    """

    if isinstance(exception, BaseSdkError):
        return exception

    exception_text = str(exception)
    is_nan_error = "(Input Error) (NaN)" in exception_text
    is_inf_error = "(Input Error) (Inf)" in exception_text
    is_input_error = "(Input Error)" in exception_text

    if is_nan_error:
        return CustomerError(
            "Unable to read data channel '{}'. Found missing (NaN) values. "
            "Please remove any missing (NaN) values in the input data.".format(channel), caused_by=exception)

    if is_inf_error:
        return CustomerError(
            "Unable to read data channel '{}'. Found infinite floating point values. "
            "Please remove any infinite floating point values in the input data.".format(channel),
            caused_by=exception)

    if is_input_error and content_type:
        return CustomerError(
            "Unable to read data channel '{}'. Requested content-type is '{}'. "
            "Please verify the data matches the requested content-type.".format(channel, content_type),
            caused_by=exception)

    if is_input_error:
        return CustomerError(
            "Unable to read data channel '{}'. "
            "Please verify the correct data channel configuration is provided.".format(channel),
            caused_by=exception)

    return exception


if sys.version_info < (3, 0):
    # `raise E, V, T` is a syntax error in Python 3, therefore using `exec`
    exec("""
def raise_with_traceback(exception, traceback=None):
    if traceback is None:
        traceback = sys.exc_info()[2]
    raise exception, None, traceback
""")
else:
    def raise_with_traceback(exception, traceback=None):
        if traceback is None:
            traceback = sys.exc_info()[2]
        raise exception.with_traceback(traceback)


class BaseSdkError(Exception):
    """Abstract base for all errors that may cause an algorithm to exit/terminate
    unsuccessfully. All direct sub-classes should be kept/maintained in this file.

    These errors are grouped into three categories:

        1. AlgorithmError: an unexpected or unknown failure that cannot be
                           avoided by the customer and is due to a bug in
                           the algorithm.

        2. CustomerError:  a failure which can be prevented/avoided by the
                           customer (e.g. change mini_batch_size).

        3. PlatformError:  a failure due to an environmental requirement not
                           being met (e.g. if the /opt/ml/training directory
                           is missing).

    All other types of errors/exceptions should be converted by default to an
    AlgorithmError.

    These classes are also responsible for providing the exit behaviour/code,
    the failure reason to output for the training service, and the log messages
    that should be printed upon termination.

    Each type of error may have multiple subclasses that inherit from both
    that error type (e.g. CustomerError) and a standard exception type
    (e.g. ValueError) to make integration easier and allow these errors to
    be caught/handled with standard handlers (instead of having SDK-specific
    error code being distributed throughout the codebase). For example, the
    following works:

    try:
        ...
        if a > 5:
            raise CustomerValueError('a should be less than 5')
        ...
    except ValueError:
        print('CustomerValueError will get handled here!')

    Args: see `Attributes` below.

    Attributes:
        message     (string): Description of why this exception was raised.
        caused_by   (exception): The underlying exception that caused this
            exception to be raised. This should be a non-BaseSdkError.
        exit_code   (int): The exit code that should be used if this exception
            makes it way to the top-level handler.
        failure_prefix (string): Prefix for the training job failure status if
            this exception is handled at the top-level. This will be seen by the
            user in the Console UI.
    """

    def __init__(self,
                 message=None,
                 caused_by=None,
                 exit_code=127,
                 failure_prefix='Algorithm Error'):
        formatted_message = BaseSdkError._format_exception_message(message, caused_by)
        super(BaseSdkError, self).__init__(formatted_message)
        self.message = formatted_message
        self.caused_by = caused_by
        self.failure_prefix = failure_prefix
        self.exit_code = exit_code

    @staticmethod
    def _format_exception_message(message, caused_by):
        """Generates the exception message.

        If a message has been explicitly passed then we use that as the exception
        message. If we also know the underlying exception type we prepend that
        to the name.

        If there is no message but we have an underlying exception then we use
        that exceptions message and prepend the type of the exception.
        """
        if message:
            formatted_message = message
        elif caused_by:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # Suppress deprecation warning
                formatted_message = getattr(caused_by, 'message', str(caused_by))
        else:
            formatted_message = "unknown error occurred"

        if caused_by:
            formatted_message += " (caused by {})".format(caused_by.__class__.__name__)

        return formatted_message

    def get_error_summary(self):
        """Return a short error summary"""
        return "{}: {}".format(self.failure_prefix, self.message)

    def get_error_detail(self):
        """Return error details"""
        return "Caused by: {}".format(self.caused_by) if self.caused_by else ""

    def _format_failure_message(self):
        message = self.get_error_summary()
        error_detail = self.get_error_detail()

        if error_detail:
            message += "\n\n{}".format(error_detail)

        return message

    def failure_message(self):
        warnings.warn("deprecated", DeprecationWarning)
        return self._format_failure_message()

    def public_failure_message(self):
        """Message to print to stdout."""
        return self._format_failure_message()

    def private_failure_message(self):
        """Message to print to the trusted error channel."""
        return self._format_failure_message()


class AlgorithmError(BaseSdkError):
    """Exception used to indicate a problem that occurred with the algorithm."""

    def __init__(self, message=None, caused_by=None):
        super(AlgorithmError, self).__init__(message,
                                             caused_by,
                                             failure_prefix='Algorithm Error',
                                             exit_code=EXIT_CODE_ALGORITHM_ERROR)


class CustomerError(BaseSdkError):
    """Exception used to indicate a problem caused by mis-configuration or other customer input."""

    def __init__(self, message=None, caused_by=None):
        super(CustomerError, self).__init__(message,
                                            caused_by,
                                            failure_prefix='Customer Error',
                                            exit_code=EXIT_CODE_CUSTOMER_ERROR)


class PlatformError(BaseSdkError):
    """Exception used to indicate a problem caused by the underlying platform (e.g. network time-outs)."""

    def __init__(self, message=None, caused_by=None):
        super(PlatformError, self).__init__(message,
                                            caused_by,
                                            failure_prefix='Platform Error',
                                            exit_code=EXIT_CODE_PLATFORM_ERROR)


class CustomerValueError(CustomerError, ValueError):
    """Exception used to indicate a problem caused by mis-configuration or other customer input."""

    def __init__(self, message=None, caused_by=None):
        super(CustomerValueError, self).__init__(message, caused_by)


class CustomerKeyError(CustomerError, KeyError):
    """Exception used to indicate a problem caused by mis-configuration or other customer input."""

    def __init__(self, message=None, caused_by=None):
        super(CustomerKeyError, self).__init__(message, caused_by)


class PlatformValueError(PlatformError, ValueError):
    """Exception used to indicate a problem caused by the underlying platform (e.g. network time-outs)."""

    def __init__(self, message=None, caused_by=None):
        super(PlatformValueError, self).__init__(message, caused_by)


class PlatformKeyError(PlatformError, KeyError):
    """Exception used to indicate a problem caused by the underlying platform (e.g. network time-outs)."""

    def __init__(self, message=None, caused_by=None):
        super(PlatformKeyError, self).__init__(message, caused_by)
