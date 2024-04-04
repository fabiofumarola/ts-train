from enum import Enum


class AggFunction(str, Enum):
    SUM = "sum"
    COUNT = "count"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    FIRST = "first"
    LAST = "last"

    def __str__(self):
        return self._value_


class GenericOperator(str, Enum):
    EQUAL = "="
    DOUBLEEQUAL = "=="
    NOTEQUAL = "!="

    def __str__(self):
        return self._value_


class NumericalOperator(str, Enum):
    LESS = "<"
    LESSEQUAL = "<="
    MORE = ">"
    MOREEQUAL = ">="

    def __str__(self):
        return self._value_


class CategoricalOperator(str, Enum):
    IN = "in"
    NOTIN = "notin"
    NOTINSPACE = "not in"

    def __str__(self):
        return self._value_


class TimeBucketGranularity(str, Enum):
    WEEK = "week"
    WEEKS = "weeks"
    DAY = "day"
    DAYS = "days"
    HOUR = "hour"
    HOURS = "hours"
    MINUTE = "minute"
    MINUTES = "minutes"
