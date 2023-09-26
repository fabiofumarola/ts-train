from typing import Union, Tuple, Optional

from pyspark.sql import DataFrame
from pyspark.sql.types import StringType
from pyspark.ml import Transformer, Model, Estimator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import (
    VectorAssembler,
    StringIndexer,
    StringIndexerModel,
    OneHotEncoder,
    OneHotEncoderModel,
)
from pyspark.ml.evaluation import (
    Evaluator,
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from xgboost.spark import SparkXGBClassifier, SparkXGBRegressor  # type: ignore


ParamValue = Union[str, int, float]
Params = dict[str, Union[list[ParamValue], ParamValue]]


def _cols_name_encode(cols_name: list[str]) -> list[str]:
    """Creates encoded columns names. This is used when encoding categorical
    variables have to be encoded.

    Args:
        cols_name (list[str]): Original columns names.

    Returns:
        list[str]: Original columns names with added "_encoded" suffix.
    """
    return [col_name + "_encoded" for col_name in cols_name]


def _cols_replace(
    df: DataFrame, temp_cols_name: list[str], cols_name: list[str]
) -> DataFrame:
    """Drops old columns (cols_name) and renames temp columns (temp_cols_name) with
    old columns names (cols_name). This is used when encoding categorical
    variables have to be encoded.

    Args:
        df (DataFrame): DataFrame on which to perform dropping and renaming.
        temp_cols_name (list[str]): columns to retain with different names
            (cols_name).
        cols_name (list[str]): columns to drop.

    Returns:
        DataFrame: DataFrame with dropped and renamed columns
    """
    df = df.drop(*cols_name)
    for temp_col_name, col_name in zip(temp_cols_name, cols_name):
        df = df.withColumnRenamed(temp_col_name, col_name)

    return df


def train_test_split(df: DataFrame, test_size=0.3) -> Tuple[DataFrame, DataFrame]:
    """Split the input DataFrame into two differnt DataFrame with the second
    test_size the total size of the original one.

    Args:
        df (DataFrame): DataFrame to split
        test_size (float, optional): Size ratio of the test DataFrame. Defaults
            to 0.3.

    Returns:
        Tuple[DataFrame, DataFrame]: train and test DataFrame in this order.
    """
    return df.randomSplit(weights=[1 - test_size, test_size])  # type: ignore


def label_encode(
    df: DataFrame,
    cols_name: list[str],
    string_indexer_model: Optional[StringIndexerModel] = None,
    order_type: str = "alphabetAsc",
) -> Tuple[DataFrame, StringIndexerModel]:
    """Encode a categorical variable considering the ordering of the options.

    Args:
        df (DataFrame): DataFrame on which to perform the encoding.
        cols_name (list[str]): Names of columns of categorical variables to label
            encode.
        string_indexer_model (Optional[StringIndexerModel], optional):
            StringIndexerModel to be used. If not provided a new one will be
            created. Defaults to None.
        order_type (str, optional): default StringIndexer is created with
            alphabetAsc as the order of the options of the categorical variable.
            Options are: frequencyDesc, frequencyAsc, alphabetDesc, alphabetAsc.
            Defaults to "alphabetAsc".

    Returns:
        Tuple[DataFrame, StringIndexerModel]: DataFrame with categorical columns
            encoded and StringIndexerModel used for the transformation.
    """
    temp_cols_name = _cols_name_encode(cols_name)

    if not string_indexer_model:
        string_indexer_model = StringIndexer(
            inputCols=cols_name,
            outputCols=temp_cols_name,
            stringOrderType=order_type,
        ).fit(df)

    df = string_indexer_model.transform(df)
    df = _cols_replace(df, temp_cols_name, cols_name)

    return df, string_indexer_model


def one_hot_encode(
    df: DataFrame,
    cols_name: list[str],
    one_hot_encoder_model: Optional[OneHotEncoderModel] = None,
    string_indexer_model: Optional[StringIndexerModel] = None,
) -> Tuple[DataFrame, OneHotEncoderModel, Optional[StringIndexerModel]]:
    """Encode a categorical variable not considering the ordering of the options.

    Args:
        df (DataFrame): DataFrame on which to perform the encoding.
        cols_name (list[str]): Names of columns of categorical variables to one-hot
            encode.
        one_hot_encoder_model (Optional[OneHotEncoderModel], optional):
            OneHotEncoderModel to be used. If not provided a new one will be
            created. Defaults to None.
        string_indexer_model (Optional[StringIndexerModel], optional):
            StringIndexerModel to be used. If not provided a new one will be
            created. Defaults to None.

    Returns:
        Tuple[DataFrame, OneHotEncoderModel, Optional[StringIndexerModel]]:
            DataFrame with categorical columns encoded, OneHotEncoderModel and
            StringIndexerModel used for the transformation.
    """
    cols_name_to_cast = []
    for col_name in cols_name:
        if isinstance(df.schema[col_name].dataType, StringType):
            cols_name_to_cast.append(col_name)
    if len(cols_name_to_cast) > 0:
        df, string_indexer_model = label_encode(
            df, cols_name_to_cast, string_indexer_model
        )

    temp_cols_name = _cols_name_encode(cols_name)

    if not one_hot_encoder_model:
        one_hot_encoder_model = OneHotEncoder(
            inputCols=cols_name, outputCols=temp_cols_name
        ).fit(df)

    df = one_hot_encoder_model.transform(df)
    df = _cols_replace(df, temp_cols_name, cols_name)

    return df, one_hot_encoder_model, string_indexer_model


def get_features_cols_name(df: DataFrame, excluded_cols_name: list[str]) -> list[str]:
    """Helper functions to retrive every feature in the DataFrame a part from the
    provided ones.

    Args:
        df (DataFrame): DataFrame on which to perform the extraction.
        excluded_cols_name (list[str]): Columns names to be excluded.

    Returns:
        list[str]: List of columns names of features present in the DataFrame.
    """
    return [col_name for col_name in df.columns if col_name not in excluded_cols_name]


def vector_assemble(
    df: DataFrame,
    features_cols_name: list[str],
    output_col_name="features",
    handle_invalid: str = "keep",
) -> DataFrame:
    """Compress features columns into only one with optimization for distributed
    training.

    Args:
        df (DataFrame): DataFrame on which to perform the transformation.
        features_cols_name (list[str]): Features columns names to be included into
            the compressed column
        output_col_name (str, optional): Column name for the output compressed
            column. Not useful to be changed. Use it only if you have a name clash.
            Defaults to "features".
        handle_invalid (str, optional): Defaults to "keep".

    Returns:
        DataFrame: DataFrame with added compressed features column.
    """
    vectorAssembler = VectorAssembler(
        inputCols=features_cols_name,
        outputCol=output_col_name,
        handleInvalid=handle_invalid,
    )

    return vectorAssembler.transform(df)


# Train/Predict/Evaluate methods
def get_estimator(
    type: str,
    features_col_name: str = "features",
    label_col_name: str = "label",
    params: Optional[Params] = None,
) -> Estimator:
    """Creates an Estimator with the params provided.

    Args:
        type (str): String to choose between classification with value "classificaiton"
            and regression with value "regression".
        features_col_name (str, optional): Column containing features compressed by
            the vector_assemble method. Defaults to "features".
        label_col_name (str, optional): Column name for the target/label of the
            dataset. Defaults to "label".
        params (Optional[Params], optional): Params passed to the model/estimator.
            Defaults to None.

    Raises:
        ValueError: with message "type parameter should be either "classification" or
            "regression"" if you provide a type param which is different from
            "classification" or "regression".

    Returns:
        Estimator: Estimator/model with provided params.
    """
    if type == "classification":
        if params is not None:
            return SparkXGBClassifier(
                features_col=features_col_name,
                label_col=label_col_name,
                enable_sparse_data_optim=True,
                missing=0.0,
                **params,  # type: ignore
            )
        else:
            return SparkXGBClassifier(
                features_col=features_col_name,
                label_col=label_col_name,
                enable_sparse_data_optim=True,
                missing=0.0,
            )
    elif type == "regression":
        if params is not None:
            return SparkXGBRegressor(
                features_col=features_col_name,
                label_col=label_col_name,
                enable_sparse_data_optim=True,
                missing=0.0,
                **params,  # type: ignore
            )
        else:
            return SparkXGBRegressor(
                features_col=features_col_name,
                label_col=label_col_name,
                enable_sparse_data_optim=True,
                missing=0.0,
            )
    else:
        raise ValueError(
            'type parameter should be either "classification" or "regression"'
        )


def evaluate_cv(
    df: DataFrame,
    estimator: Estimator,
    evaluator: Evaluator,
    num_folds: float = 3,
    parallelism: int = 1,
) -> float:
    """Evaluates the model with cross-validation and returns the avg score.

    Args:
        df (DataFrame): DataFrame on which to perform the cross-validation
        estimator (_SparkXGBEstimator): model to be trained.
        evaluator (Evaluator): estimator used for scoring
        num_folds (float, optional): number of fold for the cross-validation.
            Defaults to 3.
        parallelism (int, optional): Defaults to 1.

    Returns:
        float: avg score for the requested metric.
    """
    cv = CrossValidator(
        estimator=estimator,
        estimatorParamMaps=[None],  # type: ignore
        evaluator=evaluator,
        parallelism=parallelism,
        numFolds=num_folds,  # type: ignore
    )
    model_cv = cv.fit(df)

    return model_cv.avgMetrics[0]


def get_evaluator(
    metric: str,
    prediction_col_name: str = "prediction",
    rawPredictionCol: str = "rawPrediction",
    label_col_name: str = "label",
) -> Evaluator:
    """Creates an evaluator with the requested metric.

    Args:
        metric (str): metric to be used. Available options:
            - Classification: f1, accuracy, weightedPrecision, weightedRecall,
                weightedTruePositiveRate, weightedFalsePositiveRate,
                weightedFMeasure, truePositiveRateByLabel,
                falsePositiveRateByLabel, precisionByLabel, recallByLabel,
                fMeasureByLabel, logLoss, hammingLoss.
            - Binary classification: areaUnderROC, areaUnderPR, and those of the
                normal classification.
        prediction_col_name (str, optional): Column name of the prediction.
            Defaults to "prediction".
        rawPredictionCol (str, optional): Column name of the raw prediction.
            Defaults to "rawPrediction".
        label_col_name (str, optional): Column name for the target/label.
            Defaults to "label".

    Raises:
        ValueError: with message "Metric not supported" if you request a metric not
            supported.

    Returns:
        Evaluator: An evaluator with the requested metric.
    """
    if metric in [
        "f1",
        "accuracy",
        "weightedPrecision",
        "weightedRecall",
        "weightedTruePositiveRate",
        "weightedFalsePositiveRate",
        "weightedFMeasure",
        "truePositiveRateByLabel",
        "falsePositiveRateByLabel",
        "precisionByLabel",
        "recallByLabel",
        "fMeasureByLabel",
        "logLoss",
        "hammingLoss",
    ]:
        return MulticlassClassificationEvaluator(
            metricName=metric,  # type: ignore
            predictionCol=prediction_col_name,
            labelCol=label_col_name,
        )
    elif metric in ["areaUnderROC", "areaUnderPR"]:
        return BinaryClassificationEvaluator(
            metricName=metric,  # type: ignore
            rawPredictionCol=rawPredictionCol,
            labelCol=label_col_name,
        )
    else:
        raise ValueError("Metric not supported")


def fit(df: DataFrame, estimator: Estimator) -> Transformer:
    """Trains the estimator/model to fit the train DataFrame.

    Args:
        df (DataFrame): Train DataFrame to fit.
        estimator (_SparkXGBEstimator): Estimator/model to be used for training.

    Returns:
        Transformer: model trained.
    """
    return estimator.fit(df)


def predict(df: DataFrame, transformer: Transformer) -> DataFrame:
    """Predicts label/target for the test DataFrame with a trained model.

    Args:
        df (DataFrame): Test DataFrame on which to predict.
        transformer (Transformer): Model trained to be used for inference/prediction.

    Returns:
        DataFrame: Original DataFrame with added columns.
    """
    return transformer.transform(df)


def score(df: DataFrame, evaluator: Evaluator) -> float:
    """Given a DataFrame with labels and predictions, it uses the provided evaluator
    to score the model.

    Args:
        df (DataFrame): DataFrame on which to perform the scoring. It should have
            label and prediction columns.
        evaluator (Evaluator): Evaluator to be used for scoring.

    Returns:
        float: Value of the metric set in the Evaluator.
    """
    return evaluator.evaluate(df)


# Feature Importance
def get_features_importance(
    model: Transformer,
    features_cols_name: list[str],
    importance_type: str = "weight",
) -> dict[str, float]:
    """Provides features importance for each feature used to train the model.

    Args:
        model (Transformer): Trained model to be used.
        features_cols_name (list[str]): List of every feature used for training. Not
            only those we want the importance of.
        importance_type (str): Type of calculation used for importance:
            - weight: the number of times a feature is used to split the data across
                all trees.
            - gain: the average gain across all splits the feature is used in.
            - cover: the average coverage across all splits the feature is used in.
            - total_gain: the total gain across all splits the feature is used in.
            - total_cover: the total coverage across all splits the feature is used
                in.
            Default to "weight".

    Raises:
        ValueError: with message "features_cols_name has not the right number of
            elements"

    Returns:
        dict[str, float]: Dict with feature names as keys and feature importance as
            values.
    """
    features_ids_and_importances = model.get_feature_importances(  # type: ignore
        importance_type=importance_type
    )

    if len(features_ids_and_importances) > len(features_cols_name):
        raise ValueError("features_cols_name has not the right number of elements")

    features_names_and_importances = {}
    for idx, feature_col_name in enumerate(features_cols_name):
        feature_id = f"f{idx}"
        if feature_id in features_ids_and_importances:
            features_names_and_importances[feature_col_name] = (
                features_ids_and_importances[feature_id]
            )
        else:
            features_names_and_importances[feature_col_name] = 0.0

    return features_names_and_importances


# Params Tuning methods
# TODO allow the estimator to be regressor/classificator
def tune_parameters(
    df: DataFrame,
    params: Params,
    estimator: Estimator,
    evaluator: Evaluator,
    num_folds: int = 3,
    parallelism: int = 1,
) -> Tuple[Params, Estimator]:
    """Tunes parameters provided using cross-validation to evalute each
    configuration.

    Hyper-parameters search is performed using grid search, so pay attention to the
    number of possible options you give to each parameter.

    Args:
        df (DataFrame): DataFrame to be used as train and validation (performing
            splits).
        params (Params): dictionary where keys are param names and values are values
            for the paramter or list of possible values.
        estimator (_SparkXGBEstimator): Estimator/model to fit.
        evaluator (Evaluator): Evaluator used to score each model.
        num_folds (int, optional): Number of cross-validation folds. Defaults to 3.
        parallelism (int, optional): Defaults to 1.

    Returns:
        Tuple[Params, Model]: Optimal parameters and best model fit.
    """
    grid = ParamGridBuilder()
    for param_name, param_value in params.items():
        if isinstance(param_value, list):
            grid = grid.addGrid(estimator.getParam(param_name), param_value)
        else:
            grid = grid.baseOn((estimator.getParam(param_name), param_value))
    param_maps = grid.build()

    cv = CrossValidator(
        estimator=estimator,
        estimatorParamMaps=param_maps,
        evaluator=evaluator,
        parallelism=parallelism,
        numFolds=num_folds,
    )
    cv_model = cv.fit(df)
    best_model = cv_model.bestModel

    best_params = {}
    for param_name in params.keys():
        best_params[param_name] = best_model.getOrDefault(
            param=best_model.getParam(paramName=param_name)
        )

    return best_params, best_model
