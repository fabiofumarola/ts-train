from enum import Enum


class TimeBucketGranularity(str, Enum):
    WEEK = "week"
    WEEKS = "weeks"
    DAY = "day"
    DAYS = "days"
    HOUR = "hour"
    HOURS = "hours"
    MINUTE = "minute"
    MINUTES = "minutes"
