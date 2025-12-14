
from .core import (
    compute_quality_flags,
    compute_zero_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
    DatasetSummary,
    ColumnSummary,
)

from .viz import (
    plot_correlation_heatmap,
    plot_missing_matrix,
    plot_histograms_per_column,
    save_top_categories_tables,
)

__version__ = "0.1.0"
__all__ = [
    # core
    "compute_quality_flags",
    "compute_zero_flags",
    "correlation_matrix",
    "flatten_summary_for_print",
    "missing_table",
    "summarize_dataset",
    "top_categories",
    "DatasetSummary",
    "ColumnSummary",
    # viz
    "plot_correlation_heatmap",
    "plot_missing_matrix",
    "plot_histograms_per_column",
    "save_top_categories_tables",
]