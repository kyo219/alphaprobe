"""Built-in aggregations. Importing this package triggers registration."""

from alpharadar.aggregations._base import Aggregation, get_aggregation, register_aggregation
from alpharadar.aggregations._moving_average import MovingAverage
from alpharadar.aggregations._moving_corr import MovingCorrelation
from alpharadar.aggregations._raw import Raw

# New aggregation modules — import triggers @register_aggregation
from alpharadar.aggregations._basic_rolling import Sum, Median, Std, Var, Max, Min, Range, Skew, Kurt
from alpharadar.aggregations._rank_norm import Rank, ZScore, CoefficientOfVariation, NormalityDeviation
from alpharadar.aggregations._momentum import Momentum, RateOfChange, MeanReversion, TrendSignal
from alpharadar.aggregations._ema_family import ExponentialMovingAverage, DoubleExponentialMovingAverage, TripleExponentialMovingAverage, WeightedMovingAverage, ExponentialWeightedStd
from alpharadar.aggregations._technical import RelativeStrengthIndex, BollingerPosition, RealisedVolatility, ArchEffect
from alpharadar.aggregations._regression import LinearSlope, LinearR2
from alpharadar.aggregations._entropy import ShannonEntropy, SpectralEntropy, SampleEntropy, ApproximateEntropy, PermutationEntropy
from alpharadar.aggregations._complexity import LempelZivComplexity, HurstExponent, DetrendedFluctuationAnalysis
from alpharadar.aggregations._correlation_agg import AutoCorrelation, PartialAutoCorrelation, MutualInformation
from alpharadar.aggregations._fractional import FractionalDifference, Quantile

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
