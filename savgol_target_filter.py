from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_coeffs


SAVGOL_DIAGNOSTIC_COLUMNS = [
    "Column",
    "OutputColumn",
    "Applied",
    "Reason",
    "FitRows",
    "FitStart",
    "FitEnd",
    "CorrelationLengthSteps",
    "WindowLength",
    "Polyorder",
    "VarianceRatio",
    "PeakCountRaw",
    "PeakCountSmooth",
    "MedianPeakShiftSteps",
    "DerivativeStdRaw",
    "DerivativeStdSmooth",
    "TurningPointsRaw",
    "TurningPointsSmooth",
    "Suggestion",
]


def _nearest_odd_integer(value: float) -> int:
    lower = int(np.floor(value))
    upper = int(np.ceil(value))
    candidates = [candidate for candidate in (lower, upper, lower - 1, upper + 1) if candidate % 2 == 1 and candidate > 0]
    if not candidates:
        return 1
    return min(candidates, key=lambda candidate: (abs(candidate - value), candidate))


def _minimum_valid_window(polyorder: int) -> int:
    minimum = polyorder + 1
    if minimum % 2 == 0:
        minimum += 1
    return minimum


def estimate_correlation_length(
    values: np.ndarray,
    *,
    max_lag_steps: int | None = None,
    zero_threshold: float = 0.05,
) -> int:
    if len(values) < 3:
        return 1

    centered = values.astype(float) - float(np.nanmean(values))
    if not np.isfinite(centered).all() or np.nanstd(centered) == 0.0:
        return 1

    resolved_max_lag = max_lag_steps or min(len(centered) // 2, 512)
    resolved_max_lag = max(1, min(resolved_max_lag, len(centered) - 1))

    for lag in range(1, resolved_max_lag + 1):
        left = centered[:-lag]
        right = centered[lag:]
        denom = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denom == 0.0:
            continue
        acf_value = float(np.dot(left, right) / denom)
        if acf_value <= 0.0 or abs(acf_value) <= zero_threshold:
            return lag
    return resolved_max_lag


def choose_window_length(correlation_length_steps: int, series_length: int, polyorder: int) -> int:
    minimum_window = _minimum_valid_window(polyorder)
    available_window = series_length if series_length % 2 == 1 else series_length - 1
    if available_window < minimum_window:
        return 0

    raw_window = max(minimum_window, _nearest_odd_integer(0.5 * correlation_length_steps))
    if raw_window > available_window:
        return available_window
    return raw_window


def _count_turning_points(values: np.ndarray) -> int:
    if len(values) < 3:
        return 0
    diffs = np.diff(values)
    signs = np.sign(diffs)
    signs = signs[signs != 0]
    if len(signs) < 2:
        return 0
    return int(np.sum(signs[1:] != signs[:-1]))


def _median_peak_shift_steps(raw_peaks: np.ndarray, smooth_peaks: np.ndarray) -> float:
    if len(raw_peaks) == 0 or len(smooth_peaks) == 0:
        return float("nan")
    shifts = [int(np.min(np.abs(raw_peaks - peak))) for peak in smooth_peaks]
    return float(np.median(shifts))


def _iter_valid_segments(mask: np.ndarray) -> Iterable[tuple[int, int]]:
    start: int | None = None
    for index, is_valid in enumerate(mask):
        if is_valid and start is None:
            start = index
        elif not is_valid and start is not None:
            yield start, index
            start = None
    if start is not None:
        yield start, len(mask)


def causal_savgol_filter(values: np.ndarray, window_length: int, polyorder: int, deriv: int = 0) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype=float)
    if len(values) == 0:
        return result
    if window_length <= polyorder:
        return values.astype(float, copy=True) if deriv == 0 else result

    min_window = _minimum_valid_window(polyorder)
    coeff_cache = {
        current_window: savgol_coeffs(
            current_window,
            polyorder=polyorder,
            deriv=deriv,
            pos=current_window - 1,
            use="dot",
        )
        for current_window in range(min_window, window_length + 1, 2)
    }

    valid_mask = np.isfinite(values)
    for start, end in _iter_valid_segments(valid_mask):
        segment = values[start:end].astype(float, copy=False)
        if len(segment) < min_window:
            if deriv == 0:
                result[start:end] = segment
            continue

        for offset in range(len(segment)):
            effective_window = min(window_length, offset + 1)
            if effective_window % 2 == 0:
                effective_window -= 1
            if effective_window < min_window:
                if deriv == 0:
                    result[start + offset] = segment[offset]
                continue
            coeffs = coeff_cache[effective_window]
            result[start + offset] = float(np.dot(coeffs, segment[offset - effective_window + 1 : offset + 1]))
    return result


def _build_suggestion(
    applied: bool,
    reason: str,
    variance_ratio: float,
    peak_count_raw: int,
    peak_count_smooth: int,
) -> str:
    if not applied:
        return reason

    suggestions: list[str] = []
    if np.isfinite(variance_ratio):
        if variance_ratio < 0.7:
            suggestions.append("decrease window_length")
        elif variance_ratio > 0.95:
            suggestions.append("increase window_length")

    if peak_count_raw > 0 and peak_count_smooth < peak_count_raw:
        suggestions.append("increase polyorder or decrease window_length if peaks look flattened")
    elif peak_count_smooth > peak_count_raw:
        suggestions.append("increase window_length if noise remains")

    if not suggestions:
        return "keep"
    return "; ".join(dict.fromkeys(suggestions))


def apply_leak_safe_savgol_targets(
    df: pd.DataFrame,
    *,
    target_columns: Iterable[str],
    time_col: str = "Time",
    fit_end: str | pd.Timestamp | None = None,
    polyorder: int = 2,
    max_lag_steps: int | None = None,
    zero_threshold: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if polyorder not in {2, 3}:
        raise ValueError("polyorder must be 2 or 3 for the signal-preserving Savitzky-Golay recipe.")
    if time_col not in df.columns:
        raise KeyError(f"Missing required time column {time_col!r}.")

    result = df.copy()
    reports: list[dict[str, object]] = []
    time_values = pd.to_datetime(result[time_col], utc=True, errors="coerce")
    fit_end_ts = None if fit_end is None else pd.Timestamp(fit_end)
    if fit_end_ts is not None and fit_end_ts.tzinfo is None:
        fit_end_ts = fit_end_ts.tz_localize("UTC")
    elif fit_end_ts is not None:
        fit_end_ts = fit_end_ts.tz_convert("UTC")

    for column in target_columns:
        output_column = f"{column}_savgol"
        report = {
            "Column": column,
            "OutputColumn": output_column,
            "Applied": False,
            "Reason": "column not found",
            "FitRows": 0,
            "FitStart": pd.NaT,
            "FitEnd": pd.NaT,
            "CorrelationLengthSteps": 0,
            "WindowLength": 0,
            "Polyorder": polyorder,
            "VarianceRatio": float("nan"),
            "PeakCountRaw": 0,
            "PeakCountSmooth": 0,
            "MedianPeakShiftSteps": float("nan"),
            "DerivativeStdRaw": float("nan"),
            "DerivativeStdSmooth": float("nan"),
            "TurningPointsRaw": 0,
            "TurningPointsSmooth": 0,
            "Suggestion": "column not found",
        }

        if column not in result.columns:
            reports.append(report)
            continue

        result[output_column] = result[column].astype(float, copy=True)
        fit_mask = time_values.notna()
        if fit_end_ts is not None:
            fit_mask &= time_values <= fit_end_ts

        fit_series = result.loc[fit_mask, column].astype(float)
        fit_valid = fit_series.dropna()
        if not fit_valid.empty:
            report["FitRows"] = int(len(fit_valid))
            fit_times = time_values.loc[fit_valid.index]
            report["FitStart"] = pd.Timestamp(fit_times.iloc[0])
            report["FitEnd"] = pd.Timestamp(fit_times.iloc[-1])

        if len(fit_valid) < _minimum_valid_window(polyorder):
            report["Reason"] = "insufficient pre-evaluation history"
            report["Suggestion"] = report["Reason"]
            reports.append(report)
            continue

        fit_values = fit_valid.to_numpy(dtype=float)
        correlation_length_steps = estimate_correlation_length(
            fit_values,
            max_lag_steps=max_lag_steps,
            zero_threshold=zero_threshold,
        )
        window_length = choose_window_length(correlation_length_steps, len(fit_values), polyorder)
        report["CorrelationLengthSteps"] = int(correlation_length_steps)
        report["WindowLength"] = int(window_length)

        if window_length <= polyorder:
            report["Reason"] = "series too short for chosen polyorder"
            report["Suggestion"] = report["Reason"]
            reports.append(report)
            continue

        full_values = result[column].to_numpy(dtype=float)
        smoothed = causal_savgol_filter(full_values, window_length, polyorder, deriv=0)
        derivative = causal_savgol_filter(full_values, window_length, polyorder, deriv=1)
        result[output_column] = smoothed

        fit_smoothed = result.loc[fit_mask, output_column].dropna().to_numpy(dtype=float)
        if len(fit_smoothed) == 0:
            report["Reason"] = "filter did not produce valid output"
            report["Suggestion"] = report["Reason"]
            reports.append(report)
            continue

        raw_for_metrics = fit_values[-len(fit_smoothed) :]
        variance_raw = float(np.var(raw_for_metrics))
        variance_ratio = float(np.var(fit_smoothed) / variance_raw) if variance_raw > 0 else float("nan")
        raw_peaks, _ = find_peaks(raw_for_metrics)
        smooth_peaks, _ = find_peaks(fit_smoothed)
        raw_derivative = np.diff(raw_for_metrics)
        smooth_derivative_values = pd.Series(derivative, index=result.index).loc[fit_mask].dropna().to_numpy(dtype=float)

        report["Applied"] = True
        report["Reason"] = "ok"
        report["VarianceRatio"] = variance_ratio
        report["PeakCountRaw"] = int(len(raw_peaks))
        report["PeakCountSmooth"] = int(len(smooth_peaks))
        report["MedianPeakShiftSteps"] = _median_peak_shift_steps(raw_peaks, smooth_peaks)
        report["DerivativeStdRaw"] = float(np.std(raw_derivative)) if len(raw_derivative) > 0 else float("nan")
        report["DerivativeStdSmooth"] = (
            float(np.std(smooth_derivative_values)) if len(smooth_derivative_values) > 0 else float("nan")
        )
        report["TurningPointsRaw"] = _count_turning_points(raw_for_metrics)
        report["TurningPointsSmooth"] = _count_turning_points(fit_smoothed)
        report["Suggestion"] = _build_suggestion(
            applied=True,
            reason="ok",
            variance_ratio=variance_ratio,
            peak_count_raw=int(len(raw_peaks)),
            peak_count_smooth=int(len(smooth_peaks)),
        )
        reports.append(report)

    diagnostics = pd.DataFrame(reports)
    if diagnostics.empty:
        diagnostics = pd.DataFrame(columns=SAVGOL_DIAGNOSTIC_COLUMNS)
    else:
        diagnostics = diagnostics.reindex(columns=SAVGOL_DIAGNOSTIC_COLUMNS)
    return result, diagnostics
