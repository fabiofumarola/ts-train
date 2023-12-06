from typing import Union, Tuple, Optional
import itertools
import math
import sys

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.ml import Transformer, Estimator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.linalg import DenseVector, SparseVector
from pyspark.ml.feature import (
    Bucketizer,
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
    RegressionEvaluator,
)
from xgboost.spark import SparkXGBClassifier, SparkXGBRegressor  # type: ignore
import plotly.express as px  # type: ignore

from ts_train.common.utils import check_not_empty_dataframe, check_cols_in_dataframe


ParamValue = Union[str, int, float]
TunableParams = dict[str, Union[list[ParamValue], ParamValue]]
Params = dict[str, ParamValue]


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

    Raises:
        ValueError: if at least one column of temp_cols_name is not part of the df
            DataFrame.
        ValueError: if at least one column of cols_name is not part of the df DataFrame.

    Returns:
        DataFrame: DataFrame with dropped and renamed columns.
    """
    check_cols_in_dataframe(df, temp_cols_name, "temp_cols_name")
    check_cols_in_dataframe(df, cols_name, "cols_name")

    df = df.drop(*cols_name)
    for temp_col_name, col_name in zip(temp_cols_name, cols_name):
        df = df.withColumnRenamed(temp_col_name, col_name)

    return df


def _compute_cols_encoding(
    df: DataFrame, cols_name: list[str], order: str = "ascending"
) -> dict[str, list[Union[str, int, float]]]:
    if order == "ascending":
        ascending = True
    elif order == "descending":
        ascending = False
    else:
        raise ValueError('order param should be "ascending" or "descending"')

    cols_encoding = {}
    for col_name in cols_name:
        categ_options_row = (
            df.select(col_name)
            .distinct()
            .orderBy(col_name, ascending=ascending)
            .collect()
        )
        for categ_option_row in categ_options_row:
            categ_option = categ_option_row[col_name]
            if col_name not in cols_encoding:
                cols_encoding[col_name] = [categ_option]
            else:
                cols_encoding[col_name].append(categ_option)

    return cols_encoding


def _bucketize_df(
    df: DataFrame, value_col_name: str, buckets_width: int = 100
) -> DataFrame:
    """Used to group samples within buckets with the same range of values.

    Args:
        df (DataFrame): containing samples to be grouped.
        value_col_name (str): column name of the value to be considered to group
            samples.
        buckets_width (int, optional): width for groups, measured in units of value.
            Defaults to 100.

    Returns:
        DataFrame: containing a new columns called "bucket"
    """
    max_value = df.agg(F.max(value_col_name).alias("max_value")).collect()[0].max_value
    splits: list[float] = list(range(0, math.ceil(max_value), buckets_width))
    splits.append(float("inf"))

    bucketizer = Bucketizer(splits=splits, inputCol=value_col_name, outputCol="bucket")
    bucketed_df = bucketizer.setHandleInvalid("keep").transform(df)

    return bucketed_df


def stratified_hist(
    df: DataFrame,
    value_col_name: str,
    buckets_width: int = 100,
    buckets_num: int = 10,
    min_bucket_idx: Optional[int] = None,
    max_bucket_idx: Optional[int] = None,
    hist_params: Optional[dict[str, Union[int, float, str, bool]]] = None,
) -> None:
    """Plots a histogram with buckets counts.

    Args:
        df (DataFrame): containing samples to be bucketed.
        value_col_name (str): column name of the value to be considered to group
            samples.
        buckets_width (int, optional): width for groups, measured in units of value.
            Defaults to 100.
        buckets_num (int, optional): Number of buckets to be shown. Defaults to 10.
        min_bucket_idx (Optional[int], optional): If specified filters the buckets to be
            shown. If it the minimum index of bucket to be shown Defaults to None.
        max_bucket_idx (Optional[int], optional): If specified filters the buckets to be
            shown. If it the maximum index of bucket to be shown Defaults to None.
        hist_params (Optional[dict[str, Union[int, float, str, bool]]]): parameters to
            be passed to the px.histogram method. Default to None
    """
    bucketed_df = _bucketize_df(
        df=df, value_col_name=value_col_name, buckets_width=buckets_width
    )

    bucketed_df = bucketed_df.groupby("bucket").count()
    if min_bucket_idx:
        bucketed_df = bucketed_df.filter(F.col("bucket") > min_bucket_idx)
    if max_bucket_idx:
        bucketed_df = bucketed_df.filter(F.col("bucket") < max_bucket_idx)
    bucketed_pdf = bucketed_df.toPandas()

    if hist_params is None:
        fig = px.histogram(bucketed_pdf, x="bucket", y="count", nbins=buckets_num)
    else:
        fig = px.histogram(
            bucketed_pdf, x="bucket", y="count", nbins=buckets_num, **hist_params
        )
    fig.show()


def train_test_split(
    df: DataFrame, test_size=0.3, seed: Optional[int] = None
) -> Tuple[DataFrame, DataFrame]:
    """Splits the input DataFrame into two different DataFrames with the second
    test_size the total size of the original one.

    Args:
        df (DataFrame): DataFrame to split.
        test_size (float, optional): Size ratio of the test DataFrame. Defaults
            to 0.3.
        seed (Optional[int], optional): if you prefer to have repredicible results set a
            seed for the random sampling used inside the method. Defaults to None.

    Raises:
        ValueError: if DataFrame is empty.
        ValueError: if test_size is equal or greater then 1.

    Returns:
        Tuple[DataFrame, DataFrame]: train and test DataFrame in this order.
    """
    check_not_empty_dataframe(df)
    if test_size >= 1:
        raise ValueError("test_size should be less than 1")

    return df.randomSplit(weights=[1 - test_size, test_size], seed=seed)  # type: ignore


def stratified_train_test_split(
    df: DataFrame,
    value_col_name: str,
    test_size: float = 0.3,
    buckets_width: int = 100,
    seed: Optional[int] = None,
) -> Tuple[DataFrame, DataFrame]:
    """Splits the input DataFrame into two different DataFrames with the second
    test_size the total size of the original one.

    This split is made using the buckets provided by the method parameters. Samples are
    sampled within these buckets to guarantee fairness.

    Args:
        df (DataFrame): DataFrame to split.
        value_col_name (str): column name of the value to be considered to group
            samples.
        test_size (float, optional): size ratio of the test DataFrame. Defaults
            to 0.3.
        buckets_width (int, optional): width for groups, measured in units of value.
            Defaults to 100.
        seed (Optional[int], optional): if you prefer to have repredicible results set a
            seed for the random sampling used inside the method. Defaults to None.

    Returns:
        DataFrame: train DataFrame with 1 - test_size the original size of the
            DataFrame.
        DataFrame: test DataFrame with test_size the original size of the DataFrame.
    """
    bucketed_df = _bucketize_df(
        df=df, value_col_name=value_col_name, buckets_width=buckets_width
    )

    fractions = {
        item.bucket: 1 - test_size
        for item in bucketed_df.select(F.col("bucket")).distinct().collect()
    }
    train_df = bucketed_df.sampleBy("bucket", fractions, seed=seed)
    test_df = bucketed_df.subtract(train_df)

    train_df = train_df.drop("bucket")
    test_df = test_df.drop("bucket")

    return train_df, test_df


def label_encode(
    df,
    cols_name: list[str],
    string_indexer_model: Optional[StringIndexerModel] = None,
    order: str = "ascending",
) -> Tuple[DataFrame, StringIndexerModel, dict[str, list[Union[str, int, float]]]]:
    """Encode a categorical variable considering the ordering of the options.

    Args:
        df (DataFrame): DataFrame on which to perform the encoding.
        cols_name (list[str]): Names of columns of categorical variables to label
            encode.
        string_indexer_model (Optional[StringIndexerModel], optional):
            StringIndexerModel to be used. If not provided a new one will be
            created. Defaults to None.
        order (str, optional): it can be "ascending" or "descending".

    Raises:
        ValueError: if the DataFrame is empty.
        ValueError: if the DataFrame doesn't contain every column of cols_name.
        ValueError: if order param is different from "ascending" or "descending".

    Returns:
        Tuple[DataFrame, StringIndexerModel, dict[str, list[Union[str, int, float]]]]:
            DataFrame with categorical columns encoded, StringIndexerModel used for the
            transformation and a dict with feature names as keys and list of str, int or
            float for the values those features can be.
    """
    check_not_empty_dataframe(df)
    check_cols_in_dataframe(df, cols_name, "cols_name")

    cols_encoding = _compute_cols_encoding(
        df=df, cols_name=cols_name, order="ascending"
    )
    temp_cols_name = _cols_name_encode(cols_name)

    if not string_indexer_model:
        if order == "ascending":
            stringOrderType = "frequencyAsc"
        elif order == "descending":
            stringOrderType = "frequencyDesc"
        else:
            raise ValueError('order param should be "ascending" or "descending"')

        string_indexer_model = StringIndexer(
            inputCols=cols_name,
            outputCols=temp_cols_name,
            stringOrderType=stringOrderType,
        ).fit(df)

    df = string_indexer_model.transform(df)
    df = _cols_replace(df, temp_cols_name, cols_name)

    return df, string_indexer_model, cols_encoding


def one_hot_encode(
    df: DataFrame,
    cols_name: list[str],
    one_hot_encoder_model: Optional[OneHotEncoderModel] = None,
    string_indexer_model: Optional[StringIndexerModel] = None,
) -> Tuple[
    DataFrame,
    OneHotEncoderModel,
    Optional[StringIndexerModel],
    dict[str, list[Union[str, int, float]]],
]:
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

    Raises:
        ValueError: if the DataFrame is empty.
        ValueError: if the DataFrame doesn't contain every column of cols_name.

    Returns:
        Tuple[DataFrame, OneHotEncoderModel, Optional[StringIndexerModel],
            dict[str, list[Union[str, int, float]]]]: DataFrame with categorical columns
            encoded, OneHotEncoderModel, StringIndexerModel used for the transformation
            and a dict with feature names as keys and list of str, int or float for the
            values those features can be.
    """
    df, string_indexer_model, cols_encoding = label_encode(
        df, cols_name, string_indexer_model
    )

    temp_cols_name = _cols_name_encode(cols_name)

    if not one_hot_encoder_model:
        one_hot_encoder_model = OneHotEncoder(
            inputCols=cols_name, outputCols=temp_cols_name
        ).fit(df)

    df = one_hot_encoder_model.transform(df)
    df = _cols_replace(df, temp_cols_name, cols_name)

    return df, one_hot_encoder_model, string_indexer_model, cols_encoding


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
    cols_encoding: Optional[dict[str, list[Union[str, int, float]]]] = None,
) -> Tuple[DataFrame, list[str]]:
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

    Raises:
        ValueError: if the DataFrame is empty.
        ValueError: if the DataFrame doesn't contain every column of features_cols_name.
        ValueError: if the DataFrame already contain output_col_name column.

    Returns:
        Tuple[DataFrame, list[str]]: DataFrame with added compressed features column.
            List of feature column names considering the column encoding derived from
            label and one-hot encoding.
    """
    check_not_empty_dataframe(df)
    check_cols_in_dataframe(df, features_cols_name, "features_cols_name")
    if output_col_name in df.columns:
        raise ValueError(f"{output_col_name} already in the DataFrame")

    vectorAssembler = VectorAssembler(
        inputCols=features_cols_name,
        outputCol=output_col_name,
        handleInvalid=handle_invalid,
    )

    encoded_features_cols_name = []
    for feature_col_name in features_cols_name:
        feature_first_element = df.select(feature_col_name).first()[  # type: ignore
            feature_col_name
        ]
        if isinstance(feature_first_element, (DenseVector, SparseVector)):
            vector_len = len(feature_first_element) + 1
            for idx in range(0, vector_len):
                if (
                    cols_encoding is not None
                    and feature_col_name in cols_encoding
                    and vector_len == len(cols_encoding[feature_col_name])
                ):
                    encoded_features_cols_name.append(
                        f"{feature_col_name}_{cols_encoding[feature_col_name][idx] }"
                    )
                else:
                    encoded_features_cols_name.append(f"{feature_col_name}_{idx}")
        else:
            encoded_features_cols_name.append(feature_col_name)

    df = vectorAssembler.transform(df)

    return df, encoded_features_cols_name


# Train/Predict/Evaluate methods
def get_estimator(
    type: str,
    features_col_name: str = "features",
    label_col_name: str = "label",
    objective: Optional[str] = None,
    num_workers: int = 1,
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
        objective (str, optional): Metric to be used to optimize the model and train
            it. It is usable only with regression: reg:squarederror,
            reg:squaredlogerror, reg:logistic, reg:pseudohubererror,
            reg:absoluteerror, reg:quantileerror. Others could be found here:
            https://xgboost.readthedocs.io/en/stable/parameter.html#learning-task-parameters
            XGBoost for Spark does not allow to modify objective for classification.
        num_workers (int): How many XGBoost workers to be used to train. Each XGBoost
            worker corresponds to one spark task. It can be also provided in the params
            dictionary. This parameter overwrites the params dictionary.
        params (Optional[Params], optional): Params passed to the model/estimator.
            Defaults to None.

    Raises:
        ValueError: with message "type parameter should be either "classification" or
            "regression"" if you provide a type param which is different from
            "classification" or "regression".

    Returns:
        Estimator: Estimator (model not fitted) with provided params.
    """
    if params is None:
        params = {}

    params["num_workers"] = num_workers

    if type == "classification":
        return SparkXGBClassifier(
            features_col=features_col_name,
            label_col=label_col_name,
            enable_sparse_data_optim=True,
            missing=0.0,
            **params,  # type: ignore
        )
    elif type == "regression":
        return SparkXGBRegressor(
            features_col=features_col_name,
            label_col=label_col_name,
            enable_sparse_data_optim=True,
            missing=0.0,
            objective=objective,
            **params,  # type: ignore
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
        df (DataFrame): DataFrame on which to perform the cross-validation.
        estimator (Estimator): estimator/model to be trained.
        evaluator (Evaluator): evaluator used for scoring.
        num_folds (float, optional): number of fold for the cross-validation.
            Defaults to 3.
        parallelism (int, optional): Defaults to 1.

    Raises:
        ValueError: if the DataFrame is empty.

    Returns:
        float: avg score for the requested metric.
    """
    check_not_empty_dataframe(df)
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
            - Regression: rmse, mse, r2, mae, var
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
    elif metric in ["rmse", "mse", "r2", "mae", "var"]:
        return RegressionEvaluator(
            metricName=metric,  # type: ignore
            predictionCol=prediction_col_name,
            labelCol=label_col_name,
        )
    else:
        raise ValueError("Metric not supported")


def fit(df: DataFrame, estimator: Estimator) -> Transformer:
    """Trains the estimator/model to fit the train DataFrame.

    Args:
        df (DataFrame): Train DataFrame to fit.
        estimator (Estimator): Estimator/model to be used for training.

    Raises:
        ValueError: if the DataFrame is empty.

    Returns:
        Transformer: transformer (model already fitted).
    """
    check_not_empty_dataframe(df)

    return estimator.fit(df)


def predict(df: DataFrame, transformer: Transformer) -> DataFrame:
    """Predicts label/target for the test DataFrame with a fitted model.

    Args:
        df (DataFrame): Test DataFrame on which to predict.
        transformer (Transformer): Model fitted to be used for inference/prediction.

    Raises:
        ValueError: if the DataFrame is empty.

    Returns:
        DataFrame: Original DataFrame with added columns.
    """
    check_not_empty_dataframe(df)

    return transformer.transform(df)


def score(df: DataFrame, evaluator: Evaluator) -> float:
    """Given a DataFrame with labels and predictions, it uses the provided evaluator
    to score the model.

    Args:
        df (DataFrame): DataFrame on which to perform the scoring. It should have
            label and prediction columns.
        evaluator (Evaluator): Evaluator to be used for scoring.

    Raises:
        ValueError: if the DataFrame is empty.

    Returns:
        float: Value of the metric set in the Evaluator.
    """
    check_not_empty_dataframe(df)

    return evaluator.evaluate(df)


# Feature Importance
def get_features_importance(
    transformer: Transformer,
    encoded_features_cols_name: list[str],
    importance_type: str = "weight",
) -> dict[str, float]:
    """Provides features importance for each feature used to train the model.

    Args:
        transformer (Transformer): Fitted model to be used.
        encoded_features_cols_name (list[str]): List of every feature column name
            after the categoricaol encoding.
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
        ValueError: if features_cols_name has not the right number of elements.
        ValueError: if importance_type is not an allowed value: "weight", "gain",
            "cover", "total_gain", "total_cover".

    Returns:
        dict[str, float]: Dict with feature names as keys and feature importance as
            values.
    """
    if importance_type not in ["weight", "gain", "cover", "total_gain", "total_cover"]:
        raise ValueError(
            f'importance_type ({importance_type}) should be: "weight", '
            '"gain", "cover", "total_gain", "total_cover"'
        )

    features_ids_and_importances = transformer.get_feature_importances(  # type: ignore
        importance_type=importance_type
    )

    features_names_and_importances = {}
    for idx, feature_col_name in enumerate(encoded_features_cols_name):
        feature_id = f"f{idx}"
        if feature_id in features_ids_and_importances:
            features_names_and_importances[feature_col_name] = (
                features_ids_and_importances[feature_id]
            )
        else:
            features_names_and_importances[feature_col_name] = 0.0

    return features_names_and_importances


# Params Tuning methods
def tune_parameters(
    train_df: DataFrame,
    val_df: DataFrame,
    params: TunableParams,
    type: str,
    label_col_name: str,
    objective: Optional[str],
    mode: str,
    num_workers: int,
    evaluator: Evaluator,
) -> Tuple[Params, Transformer, Estimator]:
    """Tunes parameters provided testing on validation DataFrame to evalute each
    configuration. Training of each model is done on training DataFrame

    Hyper-parameters search is performed using grid search, so pay attention to the
    number of possible options you give to each parameter.

    Args:
        train_df (DataFrame): DataFrame to be used to train the models.
        val_df (DataFrame): DataFrame to be used to evaluate the best model.
        params (TunableParams): dictionary where keys are param names and values are
            values for the paramter or list of possible values.
        type (str): Type of training model to be used. You have to chose between
            "classification" and "regression".
        label_col_name (str): column name of target/label in the DataFrame to be
            used to fit the model for the classification or regression task.
        objective (str): Metric to be used to optimize the model and train
            it. Available options:
            - Classification: multi:softmax, multi:softprob
            - Binary classification: binary:logistic, binary:logitraw, binary:hinge
            - Regression: reg:squarederror, reg:squaredlogerror, reg:logistic,
                reg:pseudohubererror, reg:absoluteerror, reg:quantileerror
            Others could be found here: https://xgboost.readthedocs.io/en/stable/parameter.html#learning-task-parameters
            Defaults to "multi:softmax" for classification and "reg:squarederror"
            for regression.
        mode (str): "min" or "max". If you have accuracy you would like to maximize
            the metric, so you would use "max".
        num_workers (int): How many XGBoost workers to be used to train. Each XGBoost
            worker corresponds to one spark task. It can be also provided in the params
            dictionary. This parameter overwrites the params dictionary.
        evaluator (Evaluator): Evaluator used to score each model.

    Raises:
        ValueError: if DataFrame is empty.
        ValueError: if mode parameter is different from "min" or "max".

    Returns:
        Tuple[Params, Transformer]: Optimal parameters and best trasformer (best already
            fitted model).
    """
    check_not_empty_dataframe(train_df)
    check_not_empty_dataframe(val_df)

    params_temp = {}
    for key, value in params.items():
        if isinstance(value, list):
            params_temp[key] = value
        else:
            params_temp[key] = [value]

    keys, values = zip(*params_temp.items())
    params_grid = [dict(zip(keys, v)) for v in itertools.product(*values)]

    if mode == "min":
        best_score = sys.float_info.max
    elif mode == "max":
        best_score = -1
    else:
        raise ValueError("mode parameter should be min (minimize) or max (maximize).")
    best_model = None
    best_estimator = None
    best_params = None

    for params in params_grid:
        estimator = get_estimator(
            type=type,
            label_col_name=label_col_name,
            objective=objective,
            num_workers=num_workers,
            params=params,  # type: ignore
        )

        model = fit(train_df, estimator)
        val_df = predict(val_df, model)
        model_score = score(val_df, evaluator)

        print(f"Model score: {model_score} with following params:")
        print(f"    {params}")

        if (mode == "min" and model_score < best_score) or (
            mode == "max" and model_score > best_score
        ):
            best_score = model_score  # type: ignore
            best_model = model
            best_estimator = estimator
            best_params = params

    print(f"BEST Model score: {best_score} with following params:")
    print(f"    {best_params}")

    return best_params, best_model, best_estimator  # type: ignore


def tune_parameters_cv(
    df: DataFrame,
    params: TunableParams,
    estimator: Estimator,
    evaluator: Evaluator,
    num_folds: int = 3,
    parallelism: int = 1,
) -> Tuple[Params, Transformer]:
    """Tunes parameters provided using cross-validation to evalute each
    configuration.

    Hyper-parameters search is performed using grid search, so pay attention to the
    number of possible options you give to each parameter.

    Args:
        df (DataFrame): DataFrame to be used as train and validation (performing
            splits).
        params (TunableParams): dictionary where keys are param names and values are
            values for the paramter or list of possible values.
        estimator (Estimator): Estimator/model to fit.
        evaluator (Evaluator): Evaluator used to score each model.
        num_folds (int, optional): Number of cross-validation folds. Defaults to 3.
        parallelism (int, optional): Defaults to 1.

    Raises:
        ValueError: if DataFrame is empty.

    Returns:
        Tuple[Params, Transformer]: Optimal parameters and best trasformer (best already
            fitted model).
    """
    check_not_empty_dataframe(df)

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
