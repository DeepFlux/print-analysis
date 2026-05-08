from dataclasses import dataclass, field


@dataclass
class CausalResult:
    """Base dataclass for causal analysis results across all modules.

    Args:
        treatment: Name of the treatment variable.
        outcome: Name of the outcome variable.
        ate: Average Treatment Effect (point estimate).
        ate_lower: 95% CI lower bound.
        ate_upper: 95% CI upper bound.
        p_value: P-value for the Incremental Sales estimate.
        method: Estimation method used (e.g. "DoWhy-LinearRegression").
        refutation_passed: True if all mandatory refutation tests passed.
        interpretation: Plain-English summary of the result.
        warnings: List of data or model warnings to surface to the user.
    """

    treatment: str
    outcome: str
    ate: float
    ate_lower: float
    ate_upper: float
    p_value: float
    method: str
    refutation_passed: bool
    interpretation: str
    warnings: list[str] = field(default_factory=list)
