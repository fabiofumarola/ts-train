# from loguru import logger
# from typing import Any, Dict, List
# from ts_train.step.time_bucketing import TimeBucketing
# from ts_train.step.aggregation import Aggregation
# from ts_train.step.filling import Filling
# from ts_train.step.core import AbstractPipelineStep
# from ts_train.trasformation_pipeline import TrasformationPipeline


# class PipelineBuilder:
#     def __init__(self) -> None:
#         self.factory = FactorySteps()

#     def build(self, configs: List[Dict[Any, Any]]) -> TrasformationPipeline:
#         steps = [self.factory.get_step(conf) for conf in configs]
#         pipeline = TrasformationPipeline(steps)
#         return pipeline


# class FactorySteps:
#     def __init__(self) -> None:
#         self.all_steps = {
#             "TimeBucketing": TimeBucketing,
#             "Aggregation": Aggregation,
#             "Filling": Filling,
#         }

#     def get_step(self, step_setting: Dict[Any, Any]) -> AbstractPipelineStep:
#         """ """
#         # check about the classname on step_setting

#         logger.info(f"Creating step <{step_setting['classname']}>")
#         step_class = self.all_steps.get(step_setting["classname"], None)

#         if step_class:
#             step_instance = step_class(step_setting)
#         else:
#             raise Exception(f"Step class <{step_setting['classname']}> not found")

#         return step_instance
