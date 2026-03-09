"""Built-in aggregations. Importing this package triggers registration."""

from alphaprobe.aggregations._base import Aggregation, get_aggregation, register_aggregation
from alphaprobe.aggregations._moving_average import MovingAverage
from alphaprobe.aggregations._moving_corr import MovingCorrelation
from alphaprobe.aggregations._raw import Raw

# New aggregation modules — import triggers @register_aggregation
from alphaprobe.aggregations._basic_rolling import Sum, Median, Std, Var, Max, Min, Range, Skew, Kurt
from alphaprobe.aggregations._rank_norm import Rank, ZScore, CoefficientOfVariation, NormalityDeviation
from alphaprobe.aggregations._momentum import Momentum, RateOfChange, MeanReversion, TrendSignal
from alphaprobe.aggregations._ema_family import ExponentialMovingAverage, DoubleExponentialMovingAverage, TripleExponentialMovingAverage, WeightedMovingAverage, ExponentialWeightedStd
from alphaprobe.aggregations._technical import RelativeStrengthIndex, BollingerPosition, RealisedVolatility, ArchEffect
from alphaprobe.aggregations._regression import LinearSlope, LinearR2
from alphaprobe.aggregations._entropy import ShannonEntropy, SpectralEntropy, SampleEntropy, ApproximateEntropy, PermutationEntropy
from alphaprobe.aggregations._complexity import LempelZivComplexity, HurstExponent, DetrendedFluctuationAnalysis
from alphaprobe.aggregations._correlation_agg import AutoCorrelation, PartialAutoCorrelation, MutualInformation
from alphaprobe.aggregations._fractional import FractionalDifference, Quantile

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
