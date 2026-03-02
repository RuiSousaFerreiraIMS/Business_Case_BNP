import pandas as pd
import numpy as np
from scipy import stats


def compare_numeric_by_target(df, target, variables):
    """
    Compare numeric variables between two target groups.

    Returns a summary DataFrame with:
    - mean per group
    - median per group
    - difference in means
    - p-value (t-test)
    """

    results = []

    group_0 = df[df[target] == 0]
    group_1 = df[df[target] == 1]

    for var in variables:
        if var not in df.columns:
            continue

        mean_0 = group_0[var].mean()
        mean_1 = group_1[var].mean()

        median_0 = group_0[var].median()
        median_1 = group_1[var].median()

        diff = mean_1 - mean_0

        # Independent t-test
        try:
            t_stat, p_value = stats.ttest_ind(
                group_0[var].dropna(),
                group_1[var].dropna(),
                equal_var=False
            )
        except:
            p_value = np.nan

        results.append({
            "variable": var,
            "mean_group_0": mean_0,
            "mean_group_1": mean_1,
            "median_group_0": median_0,
            "median_group_1": median_1,
            "mean_difference (1-0)": diff,
            "p_value": p_value
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("p_value")

    return results_df