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
import ast
import sys

from sagemaker_algorithm_toolkit import exceptions as exc


class Hyperparameter(object):
    """Represents a single SageMaker training job hyperparameter."""

    def __init__(
        self,
        name,
        range=None,
        dependencies=None,
        required=None,
        default=None,
        tunable=False,
        tunable_recommended_range=None,
    ):
        if required is None and default is None:
            raise exc.AlgorithmError("At least one of 'required' or 'default' must be specified.")

        self.name = name
        self.range = range
        self.dependencies = dependencies
        self.required = required
        self.default = default
        self.tunable = tunable
        self.tunable_recommended_range = tunable_recommended_range

    @property
    def type(self):
        return "FreeText"

    def parse(self, value):
        return value

    def validate_range(self, value):
        if self.range is not None and value not in self.range:
            raise exc.UserError("Hyperparameter {}: {} is not in {}".format(self.name, value, self.range))

    def validate_dependencies(self, value, dependencies):
        if self.dependencies is not None:
            self.dependencies(value, dependencies)

    def format_range(self):
        raise NotImplementedError

    def format_tunable_range(self):
        raise NotImplementedError

    def format(self):
        return_value = {
            "Name": self.name,
            "Description": self.name,
            "Type": self.type,
            "IsTunable": self.tunable,
            "IsRequired": self.required or False,
        }

        try:
            return_value["Range"] = self.format_range()
        except NotImplementedError:
            pass

        if self.default is not None:
            return_value["DefaultValue"] = str(self.default)
        return return_value


class IntegerHyperparameter(Hyperparameter):
    def __init__(self, *args, **kwargs):
        if kwargs.get("range") is None:
            raise exc.AlgorithmError("range must be specified")
        super(IntegerHyperparameter, self).__init__(*args, **kwargs)

    @property
    def type(self):
        return "Integer"

    def parse(self, value):
        return int(value)

    def format_range(self):
        min_, max_ = self.range.format_as_integer()
        return {"IntegerParameterRangeSpecification": {"MinValue": min_, "MaxValue": max_}}

    def format_tunable_range(self):
        if not self.tunable or self.tunable_recommended_range is None:
            return None

        min_, max_ = self.tunable_recommended_range.format_as_integer()
        scale = self.tunable_recommended_range.scale
        return {
            "IntegerParameterRanges": [{"MinValue": min_, "MaxValue": max_, "Name": self.name, "ScalingType": scale}]
        }


class CategoricalHyperparameter(Hyperparameter):
    def __init__(self, *args, **kwargs):
        if kwargs.get("range") is None:
            raise exc.AlgorithmError("range must be specified")
        super(CategoricalHyperparameter, self).__init__(*args, **kwargs)

    @property
    def type(self):
        return "Categorical"

    def _format_range_helper(self, range_):
        if isinstance(range_, list) or isinstance(range_, tuple):
            return range_
        return range_.format()

    def format_range(self):
        return {"CategoricalParameterRangeSpecification": {"Values": self._format_range_helper(self.range)}}

    def format_tunable_range(self):
        if not self.tunable or self.tunable_recommended_range is None:
            return None
        return {
            "CategoricalParameterRanges": [
                {"Name": self.name, "Values": self._format_range_helper(self.tunable_recommended_range)}
            ]
        }


class ContinuousHyperparameter(Hyperparameter):
    def __init__(self, *args, **kwargs):
        if kwargs.get("range") is None:
            raise exc.AlgorithmError("range must be specified")
        super(ContinuousHyperparameter, self).__init__(*args, **kwargs)

    @property
    def type(self):
        return "Continuous"

    def parse(self, value):
        return float(value)

    def format_range(self):
        min_, max_ = self.range.format_as_continuous()
        return {"ContinuousParameterRangeSpecification": {"MinValue": min_, "MaxValue": max_}}

    def format_tunable_range(self):
        if not self.tunable or self.tunable_recommended_range is None:
            return None

        min_, max_ = self.tunable_recommended_range.format_as_continuous()
        scale = self.tunable_recommended_range.scale
        return {
            "ContinuousParameterRanges": [{"Name": self.name, "MinValue": min_, "MaxValue": max_, "ScalingType": scale}]
        }


class CommaSeparatedListHyperparameter(Hyperparameter):
    def __init__(self, *args, **kwargs):
        if kwargs.get("range") is None:
            raise exc.AlgorithmError("range must be specified")
        super(CommaSeparatedListHyperparameter, self).__init__(*args, **kwargs)

    def parse(self, value):
        return value.split(",")

    def validate_range(self, value):
        if any([v not in self.range for v in value]):
            raise exc.UserError("Hyperparameter {}: value {} not in range {}".format(self.name, value, self.range))


class NestedListHyperparameter(Hyperparameter):
    def __init__(self, *args, **kwargs):
        if kwargs.get("range") is None:
            raise exc.AlgorithmError("range must be specified")
        super(NestedListHyperparameter, self).__init__(*args, **kwargs)

    def parse(self, value):
        if isinstance(value, str):
            return ast.literal_eval(value)
        elif isinstance(value, list):
            return value

    def format_range(self):
        min_, max_ = self.range.format_as_integer()
        return {"NestedParameterRangeSpecification": {"MinValue": min_, "MaxValue": max_}}

    def validate_range(self, value):
        if any([element not in self.range for outer in value for element in outer]):
            raise exc.UserError("Hyperparameter {}: value {} not in range {}".format(self.name, value, self.range))


class TupleHyperparameter(Hyperparameter):
    def __init__(self, *args, **kwargs):
        if kwargs.get("range") is None:
            raise exc.AlgorithmError("range must be specified")
        super(TupleHyperparameter, self).__init__(*args, **kwargs)

    def parse(self, value):
        if isinstance(value, str):
            return eval(value)
        elif isinstance(value, tuple):
            return value

    def format_range(self):
        return {"TupleParameterRangeSpecification": {"Values": self.range}}

    def validate_range(self, value):
        if any([element not in self.range for element in value]):
            raise exc.UserError("Hyperparameter {}: value {} not in range {}".format(self.name, value, self.range))


class Hyperparameters(object):
    def __init__(self, *hyperparameters):
        self.hyperparameters = {hyperparameter.name: hyperparameter for hyperparameter in hyperparameters}
        self.aliases = {}

    def _sort_dependencies(self, hyperparameters):
        dependencies_stack = []
        visited = {hp: False for hp in hyperparameters}

        def _visit(name, visited, stack):
            visited[name] = True
            if self.hyperparameters[name].dependencies:
                for dep in self.hyperparameters[name].dependencies:
                    if dep in visited and not visited[dep]:
                        _visit(dep, visited, stack)
            stack.insert(0, name)

        for hp in hyperparameters:
            if not visited[hp]:
                _visit(hp, visited, dependencies_stack)

        return dependencies_stack

    def declare_alias(self, key_name, alias_name):
        if key_name not in self.hyperparameters:
            raise exc.AlgorithmError("Key name {}: does not exist in list of hyperparameters".format(key_name))

        self.aliases[alias_name] = key_name

    def _replace_aliases(self, user_hyperparameters):
        tmp_user_hyperparamers = {}
        for hp, value in user_hyperparameters.items():
            if hp in self.aliases:
                hp = self.aliases.get(hp)
            tmp_user_hyperparamers[hp] = value

        return tmp_user_hyperparamers

    def validate(self, user_hyperparameters):
        # Note: 0. Replace aliases with original keys
        user_hyperparameters = self._replace_aliases(user_hyperparameters)

        # NOTE: 1. Validate required or fill in default.
        for hp in self.hyperparameters:
            if hp not in user_hyperparameters:
                if self.hyperparameters[hp].required:
                    raise exc.UserError("Missing required hyperparameter: {}".format(hp))
                elif self.hyperparameters[hp].default is not None:
                    user_hyperparameters[hp] = self.hyperparameters[hp].default

        # NOTE: 2. Convert hyperparameters.
        converted_hyperparameters = {}
        for hp, value in user_hyperparameters.items():
            try:
                hyperparameter_obj = self.hyperparameters[hp]
            except KeyError:
                raise exc.UserError("Extraneous hyperparameter found: {}".format(hp))
            try:
                converted_hyperparameters[hp] = hyperparameter_obj.parse(value)
            except ValueError as e:
                raise exc.UserError("Hyperparameter {}: could not parse value".format(hp), caused_by=e)

        # NOTE: 3. Validate range.
        for hp, value in converted_hyperparameters.items():
            try:
                self.hyperparameters[hp].validate_range(value)
            except exc.UserError:
                raise
            except Exception as e:
                raise exc.AlgorithmError(
                    "Hyperparameter {}: unexpected failure when validating {}".format(hp, value), caused_by=e
                )

        # NOTE: 4. Validate dependencies.
        sorted_deps = self._sort_dependencies(converted_hyperparameters.keys())
        new_validated_hyperparameters = {}
        while sorted_deps:
            hp = sorted_deps.pop()
            value = converted_hyperparameters[hp]
            if self.hyperparameters[hp].dependencies:
                dependencies = {
                    hp_d: new_validated_hyperparameters[hp_d]
                    for hp_d in self.hyperparameters[hp].dependencies
                    if hp_d in new_validated_hyperparameters
                }
                self.hyperparameters[hp].validate_dependencies(value, dependencies)
            new_validated_hyperparameters[hp] = value

        return new_validated_hyperparameters

    def __getitem__(self, name):
        return self.hyperparameters[name]

    def format(self):
        return [hyperparameter.format() for name, hyperparameter in self.hyperparameters.items()]


class Range:
    """Abstract interface for Hyperparameter.range objects."""

    def __contains__(self, value):
        raise NotImplementedError

    def format(self):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class Interval(Range):
    LINEAR_SCALE = "Linear"

    def __init__(self, min_open=None, min_closed=None, max_open=None, max_closed=None, scale=None):
        if min_open is not None and min_closed is not None:
            raise exc.AlgorithmError("Only one of min_open and min_closed can be set")
        if max_open is not None and max_closed is not None:
            raise exc.AlgorithmError("Only one of max_open and max_closed can be set")

        self.min_open = min_open
        self.min_closed = min_closed
        self.max_open = max_open
        self.max_closed = max_closed
        self.scale = scale

    def __str__(self):
        s = ""
        if self.min_open is not None:
            s += "({}, ".format(self.min_open)
        elif self.min_closed is not None:
            s += "[{}, ".format(self.min_closed)
        else:
            s += "(-inf, "

        if self.max_open is not None:
            s += "{})".format(self.max_open)
        elif self.max_closed is not None:
            s += "{}]".format(self.max_closed)
        else:
            s += "+inf)"

        return s

    def __contains__(self, value):
        return not (
            (self.min_open is not None and value <= self.min_open)
            or (self.min_closed is not None and value < self.min_closed)
            or (self.max_open is not None and value >= self.max_open)
            or (self.max_closed is not None and value > self.max_closed)
        )

    def _format_range_value(self, open_, closed, default):
        return str(open_ if open_ is not None else closed if closed is not None else default)

    def format_as_integer(self):
        max_neg_signed_int = -(2 ** 31)
        max_signed_int = 2 ** 31 - 1
        return (
            self._format_range_value(self.min_open, self.min_closed, max_neg_signed_int),
            self._format_range_value(self.max_open, self.max_closed, max_signed_int),
        )

    def format_as_continuous(self):
        max_float = sys.float_info.max
        return (
            self._format_range_value(self.min_open, self.min_closed, -max_float),
            self._format_range_value(self.max_open, self.max_closed, max_float),
        )


class range_validator:
    """Function decorator helper to override hyperparameter's range validation."""

    def __init__(self, range):
        self.range = range

    def __call__(self, f):
        class inner(Range):
            def format(self_):
                return self.range

            def __str__(self_):
                return str(self.range)

            def __contains__(self_, value):
                return f(self.range, value)

        return inner()


class dependencies_validator:
    """Function decorator helper to override hyperparameter's dependency validation."""

    def __init__(self, dependencies):
        self.dependencies = dependencies

    def __call__(self, f):
        class inner:
            def __init__(self_):
                self_.dependencies = self.dependencies

            def __iter__(self):
                return iter(self.dependencies)

            def __next__(self):
                return next(self.dependencies)

            def __call__(self, value, dependencies):
                return f(value, dependencies)

        return inner()
