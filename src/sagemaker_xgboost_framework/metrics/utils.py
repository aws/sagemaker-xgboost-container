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


class Preconditions(object):
    CHECK_LENGTH_DEFAULT_ERROR_STRING = "Precondition check failed: length is zero."
    CHECK_NONE_DEFAULT_ERROR_STRING = "Precondition check failed: value is not None."
    CHECK_NOT_NONE_DEFAULT_ERROR_STRING = "Precondition check failed: value is None."
    CHECK_NOT_ALLOWED_VALUE_ERROR_STRING = "Precondition check failed: value is not one of the allowed values."
    CHECK_NOT_FLOAT_VALUE_ERROR_STRING = "Precondition check failed: value is not a float."

    @classmethod
    def check_length(cls, value, msg=None, exception_cls=ValueError):
        """
        Checks the length of the value. If length is zero, raises exception of type exception_cls.
        If the value is None, the exception raised by len() is re-raised as exception of type exception_cls.
        Otherwise returns the value.

        :param value: string or list value to be checked
        :param msg: the message to pass in the raised error. If None
        :param exception_cls: exception class
        :return: value or raises error when check fails.
        """

        try:
            if len(value) == 0:
                raise exception_cls(msg or cls.CHECK_LENGTH_DEFAULT_ERROR_STRING)
        except TypeError as err:
            raise exception_cls(msg or str(err))
        return value

    @classmethod
    def check_none(cls, value, msg=None, exception_cls=ValueError):
        """
        Checks if the value is None. If not None, raises exception of type exception_cls.

        :param value: value to be checked
        :param msg: the message to pass in the raised error, if None.
        :param exception_cls: exception class
        """

        if value is not None:
            raise exception_cls(msg or cls.CHECK_NONE_DEFAULT_ERROR_STRING)

    @classmethod
    def check_not_none(cls, value, msg=None, exception_cls=ValueError):
        """
        Checks if the value is None. If None, raises exception of type exception_cls.

        :param value: value to be checked
        :param msg: the message to pass in the raised error, if None.
        :param exception_cls: exception class
        :return: value or raises error when check fails.
        """

        if value is None:
            raise exception_cls(msg or cls.CHECK_NOT_NONE_DEFAULT_ERROR_STRING)
        return value

    @classmethod
    def check_allowed_value(cls, value, allowed_values, msg=None, exception_cls=ValueError):
        """
        Checks if the value is one of allowed_values. It raises exception of type exception_cls if
        allowed_values is not iterable or the value is not one of the allowed_values.

        :param value: value to be checked
        :param allowed_values: iterable of allowed values to be searched.
        :param msg: the message to pass in the raised error, if not found in the allowed_values.
        :param exception_cls: exception class
        :return: value or raises error when check fails.
        """
        try:
            if value in allowed_values:
                return value
            else:
                raise TypeError
        except TypeError:
            raise exception_cls(msg or cls.CHECK_NOT_ALLOWED_VALUE_ERROR_STRING)

    @classmethod
    def check_float(cls, value, msg=None, exception_cls=ValueError):
        try:
            float(value)
            return value
        except ValueError:
            raise exception_cls(msg or cls.CHECK_NOT_FLOAT_VALUE_ERROR_STRING)
