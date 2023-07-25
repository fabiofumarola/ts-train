from enum import Enum


class AvailableAggFunctions(str, Enum):
    SUM = "sum"
    COUNT = "count"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    FIRST = "first"
    LAST = "last"


class TimeBucketGranularity(str, Enum):
    WEEK = "week"
    WEEKS = "weeks"
    DAY = "day"
    DAYS = "days"
    HOUR = "hour"
    HOURS = "hours"
    MINUTE = "minute"
    MINUTES = "minutes"
