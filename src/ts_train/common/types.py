from typing import Annotated
import annotated_types

from pydantic.types import Strict


PositiveStrictInt = Annotated[int, Strict, annotated_types.Gt(0)]
