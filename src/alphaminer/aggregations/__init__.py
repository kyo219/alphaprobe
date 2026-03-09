"""Built-in aggregations. Importing this package triggers registration."""

from alphaminer.aggregations._base import Aggregation, get_aggregation, register_aggregation
from alphaminer.aggregations._moving_average import MovingAverage
from alphaminer.aggregations._moving_corr import MovingCorrelation
from alphaminer.aggregations._raw import Raw

# New aggregation modules — import triggers @register_aggregation
from alphaminer.aggregations._basic_rolling import Sum, Median, Std, Var, Max, Min, Range, Skew, Kurt
from alphaminer.aggregations._rank_norm import Rank, ZScore, CoefficientOfVariation, NormalityDeviation
from alphaminer.aggregations._momentum import Momentum, RateOfChange, MeanReversion, TrendSignal
from alphaminer.aggregations._ema_family import ExponentialMovingAverage, DoubleExponentialMovingAverage, TripleExponentialMovingAverage, WeightedMovingAverage, ExponentialWeightedStd
from alphaminer.aggregations._technical import RelativeStrengthIndex, BollingerPosition, RealisedVolatility, ArchEffect
from alphaminer.aggregations._regression import LinearSlope, LinearR2
from alphaminer.aggregations._entropy import ShannonEntropy, SpectralEntropy, SampleEntropy, ApproximateEntropy, PermutationEntropy
from alphaminer.aggregations._complexity import LempelZivComplexity, HurstExponent, DetrendedFluctuationAnalysis
from alphaminer.aggregations._correlation_agg import AutoCorrelation, PartialAutoCorrelation, MutualInformation
from alphaminer.aggregations._fractional import FractionalDifference, Quantile

__all__ = [
    "Aggregation",
    "get_aggregation",
    "register_aggregation",
    "MovingAverage",
    "MovingCorrelation",
    "Raw",
    "Sum", "Median", "Std", "Var", "Max", "Min", "Range", "Skew", "Kurt",
    "Rank", "ZScore", "CoefficientOfVariation", "NormalityDeviation",
    "Momentum", "RateOfChange", "MeanReversion", "TrendSignal",
    "ExponentialMovingAverage", "DoubleExponentialMovingAverage", "TripleExponentialMovingAverage", "WeightedMovingAverage", "ExponentialWeightedStd",
    "RelativeStrengthIndex", "BollingerPosition", "RealisedVolatility", "ArchEffect",
    "LinearSlope", "LinearR2",
    "ShannonEntropy", "SpectralEntropy", "SampleEntropy", "ApproximateEntropy", "PermutationEntropy",
    "LempelZivComplexity", "HurstExponent", "DetrendedFluctuationAnalysis",
    "AutoCorrelation", "PartialAutoCorrelation", "MutualInformation",
    "FractionalDifference", "Quantile",
]
