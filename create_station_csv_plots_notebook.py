from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


def md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).strip("\n").splitlines(keepends=True),
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip("\n").splitlines(keepends=True),
    }


NOTEBOOK_PATH = Path("Ionosphere_station_csv_plots_2025.ipynb")


cells = [
    md_cell(
        """
        # Ionosphere Station CSV Plots

        This notebook reproduces the main graph types from `Ionosphere_optuna_pipeline_2025_.ipynb`,
        but it reads already-exported station CSV files instead of rerunning the full pipeline.

        Edit `station_code` and `selected_train_days` in the parameter cell, then run the notebook.

        Note:
        - The exported station CSV bundles currently contain only Optuna ML model outputs.
        - IRI / analytic baseline traces from the original notebook are shown only if those models are present in the CSVs.
        """
    ),
    code_cell(
        """
        import json
        from pathlib import Path

        import pandas as pd
        import plotly.graph_objects as go
        from plotly.colors import qualitative
        from plotly.subplots import make_subplots
        from IPython.display import display

        try:
            from ionosphere_pipeline import (
                FEATURE_COLORS,
                LINE_STYLES,
                LINE_WIDTHS,
                MODEL_COLORS,
                MODEL_ORDER,
                PATTERN_SHAPES,
                PLOTLY_TEMPLATE,
                SEASON_COLORS,
                TARGET_TITLES,
            )
        except ImportError:
            FEATURE_COLORS = {}
            LINE_STYLES = {
                "ElasticNet": "solid",
                "RandomForest": "solid",
                "XGBoost": "solid",
                "IRI": "dash",
                "Analytic": "dot",
            }
            LINE_WIDTHS = {
                "ElasticNet": 2.0,
                "RandomForest": 2.0,
                "XGBoost": 2.0,
                "IRI": 2.0,
                "Analytic": 2.0,
            }
            MODEL_COLORS = {
                "ElasticNet": "#2E86AB",
                "RandomForest": "#D1495B",
                "XGBoost": "#2A9D8F",
                "IRI": "#6C757D",
                "Analytic": "#8E5EA2",
            }
            MODEL_ORDER = ["ElasticNet", "RandomForest", "XGBoost", "IRI", "Analytic"]
            PATTERN_SHAPES = {
                "ElasticNet": "",
                "RandomForest": "/",
                "XGBoost": "x",
                "IRI": ".",
                "Analytic": "\\\\",
            }
            PLOTLY_TEMPLATE = "plotly_white"
            SEASON_COLORS = {
                "Winter 24/25": "rgba(156, 197, 255, 0.22)",
                "Spring 2025": "rgba(176, 230, 124, 0.20)",
                "Summer 2025": "rgba(255, 217, 102, 0.20)",
                "Autumn 2025": "rgba(244, 177, 131, 0.20)",
            }
            TARGET_TITLES = {"foF2": "foF2", "MUFD": "MUFD"}

        BASE_DIR = Path(".")
        DEFAULT_FEATURE_PALETTE = qualitative.Dark24 + qualitative.Safe + qualitative.Set3
        DEFAULT_SEASON_ORDER = ["Winter 24/25", "Spring 2025", "Summer 2025", "Autumn 2025"]
        """
    ),
    code_cell(
        """
        def _pick_latest(paths):
            paths = sorted(paths, key=lambda path: (path.stat().st_mtime, path.name), reverse=True)
            return paths[0] if paths else None


        def _filter_root_exports(paths, station_code, allow_window=False):
            blocked = {"conflicted copy"}
            if not allow_window:
                blocked.update({f"{station_code}_window_", "_window_"})
            result = []
            for path in paths:
                lower = path.name.lower()
                if any(token.lower() in lower for token in blocked):
                    continue
                result.append(path)
            return result


        def _candidate_export_dirs(base_dir: Path, station_code: str) -> list[Path]:
            dirs = []
            station_results_dir = base_dir / "results" / station_code
            if station_results_dir.exists():
                dirs.append(station_results_dir)
            dirs.append(base_dir)
            return dirs


        def _resolve_from_dirs(base_dir: Path, station_code: str, patterns):
            if isinstance(patterns, str):
                patterns = [patterns]

            for pattern in patterns:
                for directory in _candidate_export_dirs(base_dir, station_code):
                    candidates = list(directory.glob(pattern))
                    candidates = _filter_root_exports(
                        candidates,
                        station_code,
                        allow_window="_window_" in pattern.lower(),
                    )
                    chosen = _pick_latest(candidates)
                    if chosen is not None:
                        return chosen
            return None


        def resolve_station_exports(base_dir: Path, station_code: str, preferred_train_days=None) -> dict:
            station_code = station_code.upper().strip()
            window_prefix = f"{station_code}_window_{preferred_train_days}d_" if preferred_train_days else None

            def with_window_fallback(main_pattern: str, suffix: str):
                patterns = [main_pattern]
                if window_prefix:
                    patterns.append(f"{window_prefix}{suffix}")
                return patterns

            patterns = {
                "preds": with_window_fallback(
                    f"{station_code}_*predicted_time_series*.csv",
                    "*predicted_time_series*.csv",
                ),
                "metrics": with_window_fallback(
                    f"{station_code}_*optuna_daily*.csv",
                    "*optuna_daily*.csv",
                ),
                "summary": with_window_fallback(
                    f"{station_code}_*summary*.csv",
                    "*summary*.csv",
                ),
                "trials": with_window_fallback(
                    f"{station_code}_*optuna_trials*.csv",
                    "*optuna_trials*.csv",
                ),
                "shap": with_window_fallback(
                    f"{station_code}_*shap_optuna*.csv",
                    "*shap_optuna*.csv",
                ),
                "phys": f"{station_code}_*phys_daily*.csv",
            }

            resolved = {}
            for key, pattern in patterns.items():
                resolved[key] = _resolve_from_dirs(base_dir, station_code, pattern)

            metadata_candidates = [
                base_dir / f"{station_code}.json",
                base_dir / f"ml_pipeline_runner.{station_code}.json",
            ]
            resolved["metadata"] = next((path for path in metadata_candidates if path.exists()), None)

            missing = [key for key in ("preds", "metrics", "trials", "shap") if resolved[key] is None]
            if missing:
                raise FileNotFoundError(
                    f"Missing export files for station {station_code}: {', '.join(missing)}"
                )

            return resolved


        def load_station_bundle(base_dir: Path, station_code: str, preferred_train_days=None) -> dict:
            files = resolve_station_exports(base_dir, station_code, preferred_train_days=preferred_train_days)

            preds_df = pd.read_csv(files["preds"])
            metrics_df = pd.read_csv(files["metrics"])
            trials_df = pd.read_csv(files["trials"])
            shap_df = pd.read_csv(files["shap"])
            summary_df = pd.read_csv(files["summary"]) if files["summary"] else pd.DataFrame()
            phys_df = pd.read_csv(files["phys"]) if files["phys"] else pd.DataFrame()

            for col in ["RunDate", "SourceTime", "TargetTime"]:
                if col in preds_df.columns:
                    preds_df[col] = pd.to_datetime(preds_df[col], errors="coerce")
            for col in ["Date"]:
                if col in metrics_df.columns:
                    metrics_df[col] = pd.to_datetime(metrics_df[col], errors="coerce")
                if col in trials_df.columns:
                    trials_df[col] = pd.to_datetime(trials_df[col], errors="coerce")
                if col in shap_df.columns:
                    shap_df[col] = pd.to_datetime(shap_df[col], errors="coerce")
                if col in phys_df.columns:
                    phys_df[col] = pd.to_datetime(phys_df[col], errors="coerce")

            metadata = {}
            if files["metadata"]:
                metadata = json.loads(files["metadata"].read_text(encoding="utf-8"))

            return {
                "files": files,
                "preds_df": preds_df,
                "metrics_df": metrics_df,
                "summary_df": summary_df,
                "trials_df": trials_df,
                "shap_df": shap_df,
                "phys_df": phys_df,
                "metadata": metadata,
            }


        def get_station_context(bundle: dict, station_code: str) -> dict:
            metadata = bundle.get("metadata", {})
            station_meta = metadata.get("station_metadata", {})
            config = metadata.get("config", {})

            station_name = station_meta.get("station_name", station_code)
            latitude = station_meta.get("latitude")
            longitude = station_meta.get("longitude")

            seasons = {}
            for season_name, bounds in config.get("seasons", {}).items():
                if not bounds:
                    continue
                season_start = pd.Timestamp(bounds[0]).tz_localize(None)
                season_end = pd.Timestamp(bounds[1]).tz_localize(None)
                seasons[season_name] = (season_start, season_end)

            if not seasons:
                metrics_df = bundle["metrics_df"].dropna(subset=["Date"]).copy()
                for season_name, group in metrics_df.groupby("Season", dropna=True):
                    seasons[season_name] = (group["Date"].min(), group["Date"].max())

            season_order = [name for name in DEFAULT_SEASON_ORDER if name in seasons]
            season_order.extend([name for name in seasons if name not in season_order])

            return {
                "station_code": station_code,
                "station_name": station_name,
                "latitude": latitude,
                "longitude": longitude,
                "season_order": season_order,
                "seasons": seasons,
            }


        def add_season_bands(fig, seasons: dict, season_order: list[str], rows=None, annotation_row=1):
            row_values = None
            if rows is None:
                row_values = [None]
            elif isinstance(rows, int):
                row_values = list(range(1, rows + 1))
            else:
                row_values = list(rows)

            for season_name in season_order:
                if season_name not in seasons:
                    continue
                season_start, season_end = seasons[season_name]
                season_color = SEASON_COLORS.get(season_name, "rgba(200, 200, 200, 0.15)")

                for row in row_values:
                    kwargs = dict(
                        x0=season_start,
                        x1=season_end,
                        fillcolor=season_color,
                        opacity=1,
                        layer="below",
                        line_width=0,
                    )
                    if row is None:
                        fig.add_vrect(**kwargs)
                    else:
                        fig.add_vrect(**kwargs, row=row, col=1)

                if annotation_row is None:
                    continue

                annotation_kwargs = dict(
                    x=season_start + (season_end - season_start) / 2,
                    text=season_name,
                    showarrow=False,
                    font=dict(size=9, color="#444"),
                    y=1,
                    yref="paper",
                    yanchor="bottom",
                )
                if rows is None:
                    fig.add_annotation(**annotation_kwargs)
                else:
                    fig.add_annotation(**annotation_kwargs, row=annotation_row, col=1)


        def build_observed_series(preds_df: pd.DataFrame, year: int, series_kind: str = "original") -> pd.DataFrame:
            value_columns = [f"foF2_{series_kind}", f"MUFD_{series_kind}"]
            keep_columns = ["TargetTime"] + [col for col in value_columns if col in preds_df.columns]
            overview_df = preds_df[keep_columns].copy()
            overview_df["TargetTime"] = pd.to_datetime(overview_df["TargetTime"], errors="coerce")
            overview_df = overview_df.dropna(subset=["TargetTime"])
            overview_df = overview_df[overview_df["TargetTime"].dt.year == year]
            overview_df = overview_df.groupby("TargetTime", as_index=False).first().sort_values("TargetTime")
            return overview_df


        def build_season_summary(metrics_df: pd.DataFrame, train_days: int) -> pd.DataFrame:
            subset = metrics_df[metrics_df["TrainDays"] == train_days].copy()
            if subset.empty:
                return subset
            summary = (
                subset.groupby(["Season", "Target", "Model"], as_index=False)[["R2", "MAE", "RMSE", "MAPE_%", "CC"]]
                .mean()
                .round(4)
            )
            return summary


        def build_phys_season_summary(phys_df: pd.DataFrame, train_days: int) -> pd.DataFrame:
            if phys_df.empty:
                return phys_df
            subset = phys_df[phys_df["TrainDays"] == train_days].copy()
            if subset.empty:
                return subset
            summary = (
                subset.groupby(["Season", "Target", "Model"], as_index=False)[["R2", "MAE"]]
                .mean()
                .round(4)
            )
            return summary


        def build_period_metric_table(
            metrics_df: pd.DataFrame,
            train_days: int,
            range_kind: str = "week",
            period_labels=None,
            date_start=None,
            date_end=None,
        ) -> pd.DataFrame:
            subset = metrics_df[metrics_df["TrainDays"] == train_days].copy()
            subset = subset.dropna(subset=["Date"])
            if date_start:
                subset = subset[subset["Date"] >= pd.Timestamp(date_start)]
            if date_end:
                subset = subset[subset["Date"] <= pd.Timestamp(date_end)]
            if subset.empty:
                return subset

            if range_kind == "season":
                subset["RangeLabel"] = subset["Season"]
                subset["PeriodStart"] = subset.groupby("Season")["Date"].transform("min")
                subset["PeriodEnd"] = subset.groupby("Season")["Date"].transform("max")
            elif range_kind == "month":
                period = subset["Date"].dt.to_period("M")
                subset["RangeLabel"] = period.astype(str)
                subset["PeriodStart"] = period.dt.start_time
                subset["PeriodEnd"] = period.dt.end_time
            elif range_kind == "week":
                iso = subset["Date"].dt.isocalendar()
                subset["RangeLabel"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
                subset["PeriodStart"] = subset["Date"] - pd.to_timedelta(subset["Date"].dt.weekday, unit="D")
                subset["PeriodEnd"] = subset["PeriodStart"] + pd.Timedelta(days=6)
            elif range_kind == "day":
                subset["RangeLabel"] = subset["Date"].dt.strftime("%Y-%m-%d")
                subset["PeriodStart"] = subset["Date"].dt.floor("D")
                subset["PeriodEnd"] = subset["Date"].dt.floor("D")
            else:
                raise ValueError("range_kind must be one of: season, month, week, day")

            if period_labels:
                subset = subset[subset["RangeLabel"].isin(period_labels)]
            if subset.empty:
                return subset

            agg_spec = {
                "R2": ("R2", "mean"),
                "MAE": ("MAE", "mean"),
                "N_days": ("Date", "nunique"),
            }
            for metric_name in ["RMSE", "CC", "MAPE_%"]:
                if metric_name in subset.columns:
                    agg_spec[metric_name] = (metric_name, "mean")

            grouped = (
                subset.groupby(["RangeLabel", "PeriodStart", "PeriodEnd", "Target", "Model"], as_index=False)
                .agg(**agg_spec)
                .sort_values(["PeriodStart", "Target", "Model"])
            )
            return grouped


        def aggregate_prediction_frame(
            preds_df: pd.DataFrame,
            target: str,
            train_days: int,
            year: int,
            resample_freq=None,
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
            subset = preds_df[(preds_df["Target"] == target) & (preds_df["TrainDays"] == train_days)].copy()
            subset["Time"] = pd.to_datetime(subset["TargetTime"], errors="coerce")
            subset = subset.dropna(subset=["Time"])
            subset = subset[subset["Time"].dt.year == year]
            if subset.empty:
                return pd.DataFrame(), pd.DataFrame()

            actual_df = (
                subset.groupby("Time", as_index=False)["ActualLabel"]
                .median()
                .rename(columns={"ActualLabel": "Actual"})
                .sort_values("Time")
            )

            pred_df = (
                subset.groupby(["Time", "Model"], as_index=False)["Predicted"]
                .median()
                .sort_values(["Time", "Model"])
            )

            if resample_freq:
                actual_df = (
                    actual_df.set_index("Time")
                    .resample(resample_freq)
                    .median(numeric_only=True)
                    .reset_index()
                    .dropna()
                )
                pred_df = (
                    pred_df.set_index("Time")
                    .groupby("Model")
                    .resample(resample_freq)["Predicted"]
                    .median()
                    .reset_index()
                    .dropna()
                )

            return actual_df, pred_df


        def season_metric_annotations(pred_df: pd.DataFrame, actual_df: pd.DataFrame, seasons: dict, season_order: list[str]) -> list[dict]:
            if pred_df.empty or actual_df.empty:
                return []

            merged = pred_df.merge(actual_df, on="Time", how="inner")
            annotations = []
            for season_name in season_order:
                if season_name not in seasons:
                    continue
                season_start, season_end = seasons[season_name]
                season_slice = merged[(merged["Time"] >= season_start) & (merged["Time"] <= season_end)].copy()
                if season_slice.empty:
                    continue

                parts = []
                for model_name in MODEL_ORDER:
                    model_slice = season_slice[season_slice["Model"] == model_name]
                    if len(model_slice) < 3:
                        continue
                    actual = model_slice["Actual"]
                    predicted = model_slice["Predicted"]
                    ss_res = ((actual - predicted) ** 2).sum()
                    ss_tot = ((actual - actual.mean()) ** 2).sum()
                    r2 = float("nan") if ss_tot == 0 else 1 - ss_res / ss_tot
                    if pd.notna(r2):
                        parts.append(f"{model_name}={r2:.2f}")

                if not parts:
                    continue

                annotations.append(
                    {
                        "x": season_start + (season_end - season_start) / 2,
                        "text": " | ".join(parts),
                    }
                )
            return annotations


        def top_shap_features(shap_df: pd.DataFrame, target: str, train_days: int, top_n=None):
            subset = shap_df[(shap_df["Target"] == target) & (shap_df["TrainDays"] == train_days)].copy()
            if subset.empty:
                return []
            feature_order = (
                subset.groupby("Feature", as_index=False)["Pct_s"]
                .mean()
                .sort_values("Pct_s", ascending=False)["Feature"]
                .tolist()
            )
            return feature_order if top_n in (None, 0) else feature_order[:top_n]


        def get_feature_color(feature_name: str) -> str:
            if feature_name in FEATURE_COLORS:
                return FEATURE_COLORS[feature_name]
            palette_index = sum(ord(char) for char in feature_name) % len(DEFAULT_FEATURE_PALETTE)
            return DEFAULT_FEATURE_PALETTE[palette_index]
        """
    ),
    code_cell(
        """
        station_code = "EA036"
        selected_train_days = 21
        series_year = 2025

        comparison_range_kind = "week"   # "season", "month", "week", "day"
        comparison_metric = "R2"         # "R2" or "MAE"
        comparison_period_labels = None  # e.g. ["2025-03", "2025-04"] or ["2025-W10"] or ["2025-03-15"]
        comparison_date_start = None     # overrides config.ml_date_start when set
        comparison_date_end = None       # overrides config.ml_date_end when set

        overview_series_kind = "original"  # "original" or "filtered"
        resample_freq = None               # e.g. "D" for daily median
        shap_top_features = 12             # set to None to plot all features
        """
    ),
    code_cell(
        """
        bundle = load_station_bundle(BASE_DIR, station_code, preferred_train_days=selected_train_days)
        station = get_station_context(bundle, station_code)

        preds_df = bundle["preds_df"]
        metrics_df = bundle["metrics_df"]
        summary_df = bundle["summary_df"]
        trials_df = bundle["trials_df"]
        shap_df = bundle["shap_df"]
        phys_df = bundle["phys_df"]

        resolved_files_df = pd.DataFrame(
            {
                "kind": list(bundle["files"].keys()),
                "path": [str(path) if path else None for path in bundle["files"].values()],
            }
        )

        dataset_summary_df = pd.DataFrame(
            [
                {"item": "station_code", "value": station["station_code"]},
                {"item": "station_name", "value": station["station_name"]},
                {"item": "latitude", "value": station["latitude"]},
                {"item": "longitude", "value": station["longitude"]},
                {"item": "prediction_rows", "value": len(preds_df)},
                {"item": "metric_rows", "value": len(metrics_df)},
                {"item": "trial_rows", "value": len(trials_df)},
                {"item": "shap_rows", "value": len(shap_df)},
                {"item": "phys_rows", "value": len(phys_df)},
                {"item": "train_days_available", "value": sorted(metrics_df["TrainDays"].dropna().unique().tolist())},
                {"item": "targets_available", "value": sorted(metrics_df["Target"].dropna().unique().tolist())},
                {"item": "models_available", "value": sorted(set(metrics_df["Model"].dropna().tolist()) | set(phys_df.get("Model", pd.Series(dtype=object)).dropna().tolist()))},
            ]
        )

        display(resolved_files_df)
        display(dataset_summary_df)
        display(metrics_df.head())
        display(shap_df.head())
        """
    ),
    md_cell(
        """
        ## Graph 1. Station Time-Series Overview

        Recreates the first notebook graph type: stacked `foF2` and `MUFD` time series with season shading.
        """
    ),
    code_cell(
        """
        overview_df = build_observed_series(preds_df, year=series_year, series_kind=overview_series_kind)

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            subplot_titles=("foF2", "MUFD"),
            vertical_spacing=0.10,
        )
        add_season_bands(fig, station["seasons"], station["season_order"], rows=2)

        if "foF2_" + overview_series_kind in overview_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=overview_df["TargetTime"],
                    y=overview_df["foF2_" + overview_series_kind],
                    mode="lines",
                    name=f"foF2 ({overview_series_kind})",
                    line=dict(color="steelblue", width=1.6),
                ),
                row=1,
                col=1,
            )

        if "MUFD_" + overview_series_kind in overview_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=overview_df["TargetTime"],
                    y=overview_df["MUFD_" + overview_series_kind],
                    mode="lines",
                    name=f"MUFD ({overview_series_kind})",
                    line=dict(color="tomato", width=1.6),
                ),
                row=2,
                col=1,
            )

        fig.update_yaxes(title_text="foF2", row=1, col=1)
        fig.update_yaxes(title_text="MUFD", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            height=680,
            hovermode="x unified",
            title=f"{station['station_name']} ({station['station_code']}): observed series overview",
        )
        fig.show()
        """
    ),
    md_cell(
        """
        ## Graph 2. Seasonal R2 Bars

        Recreates the seasonal grouped-bar comparison from the original notebook, using the station daily metrics CSV.
        """
    ),
    code_cell(
        """
        season_summary_df = build_season_summary(metrics_df, train_days=selected_train_days)
        phys_season_summary_df = build_phys_season_summary(phys_df, train_days=selected_train_days)
        if not phys_season_summary_df.empty:
            season_summary_df = pd.concat([season_summary_df, phys_season_summary_df], ignore_index=True, sort=False)
        display(season_summary_df.head(20))

        if season_summary_df.empty:
            print(f"No seasonal summary rows for train_days={selected_train_days}.")
        else:
            for target in sorted(season_summary_df["Target"].unique()):
                sub = season_summary_df[season_summary_df["Target"] == target].copy()
                fig = go.Figure()

                for model_name in MODEL_ORDER:
                    model_frame = (
                        sub[sub["Model"] == model_name]
                        .set_index("Season")
                        .reindex(station["season_order"])
                        .reset_index()
                    )
                    if model_frame["R2"].isna().all():
                        continue

                    fig.add_trace(
                        go.Bar(
                            name=model_name,
                            x=model_frame["Season"],
                            y=model_frame["R2"],
                            marker_color=MODEL_COLORS.get(model_name, "#666666"),
                            marker_pattern_shape=PATTERN_SHAPES.get(model_name, ""),
                            marker_pattern_fgcolor="white",
                            opacity=0.86,
                        )
                    )

                fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1.0)
                fig.update_layout(
                    barmode="group",
                    template=PLOTLY_TEMPLATE,
                    height=430,
                    legend=dict(orientation="h", y=-0.18),
                    xaxis_title="Season",
                    yaxis_title="R2",
                    title=(
                        f"R2 by season: {target} | {station['station_name']} ({station['station_code']})"
                        f"<br><sup>train={selected_train_days} d | CSV-backed station summary</sup>"
                    ),
                )
                fig.show()
        """
    ),
    md_cell(
        """
        ## Graph 3. Period Metric Bars

        Recreates the grouped metric comparison by season/month/week/day. Change `comparison_range_kind`
        and `comparison_metric` in the parameter cell to switch the view.
        """
    ),
    code_cell(
        """
        range_metrics_df = build_period_metric_table(
            metrics_df,
            train_days=selected_train_days,
            range_kind=comparison_range_kind,
            period_labels=comparison_period_labels,
            date_start=comparison_date_start,
            date_end=comparison_date_end,
        )
        if not phys_df.empty:
            phys_range_metrics_df = build_period_metric_table(
                phys_df,
                train_days=selected_train_days,
                range_kind=comparison_range_kind,
                period_labels=comparison_period_labels,
                date_start=comparison_date_start,
                date_end=comparison_date_end,
            )
            if not phys_range_metrics_df.empty:
                range_metrics_df = pd.concat([range_metrics_df, phys_range_metrics_df], ignore_index=True, sort=False)
        display(range_metrics_df.head(20))

        if range_metrics_df.empty:
            print("No rows available for the selected period filter.")
        else:
            axis_title_map = {
                "season": "Season",
                "month": "Month",
                "week": "ISO week",
                "day": "Day",
            }
            range_label_title = axis_title_map[comparison_range_kind]
            target_values = sorted(range_metrics_df["Target"].unique())

            fig = make_subplots(
                rows=len(target_values),
                cols=1,
                shared_xaxes=False,
                subplot_titles=tuple(target_values),
                vertical_spacing=0.12,
            )

            for row_index, target in enumerate(target_values, start=1):
                target_frame = range_metrics_df[range_metrics_df["Target"] == target].copy()
                for model_name in MODEL_ORDER:
                    model_frame = target_frame[target_frame["Model"] == model_name].copy()
                    if model_frame.empty:
                        continue

                    hover_rows = list(
                        zip(
                            model_frame["N_days"],
                            model_frame["PeriodStart"].dt.strftime("%Y-%m-%d"),
                            model_frame["PeriodEnd"].dt.strftime("%Y-%m-%d"),
                        )
                    )

                    fig.add_trace(
                        go.Bar(
                            name=model_name,
                            x=model_frame["RangeLabel"],
                            y=model_frame[comparison_metric],
                            marker_color=MODEL_COLORS.get(model_name, "#666666"),
                            marker_pattern_shape=PATTERN_SHAPES.get(model_name, ""),
                            marker_pattern_fgcolor="white",
                            opacity=0.88,
                            showlegend=row_index == 1,
                            customdata=hover_rows,
                            hovertemplate=(
                                "<b>%{x}</b><br>"
                                "Model: %{fullData.name}<br>"
                                + comparison_metric + ": %{y:.4f}<br>"
                                + "Days: %{customdata[0]}<br>"
                                + "From: %{customdata[1]}<br>"
                                + "To: %{customdata[2]}<extra></extra>"
                            ),
                        ),
                        row=row_index,
                        col=1,
                    )
                    fig.update_yaxes(title_text=comparison_metric, row=row_index, col=1)
                    fig.update_xaxes(title_text=range_label_title, row=row_index, col=1)

            fig.update_layout(
                barmode="group",
                template=PLOTLY_TEMPLATE,
                height=max(420, 320 * len(target_values)),
                legend=dict(orientation="h", y=-0.18),
                title=(
                    f"{comparison_metric} by {range_label_title.lower()}: {station['station_name']} ({station['station_code']})"
                    f"<br><sup>train={selected_train_days} d | station CSV metrics</sup>"
                ),
            )
            fig.show()
        """
    ),
    md_cell(
        """
        ## Graph 4. Forecast Time Series

        Recreates the forecast-vs-actual line graph for each target. Since the station export CSVs store ML
        model predictions directly, this figure uses those model rows without rerunning the pipeline.
        """
    ),
    code_cell(
        """
        for target in sorted(preds_df["Target"].dropna().unique()):
            actual_df, pred_df = aggregate_prediction_frame(
                preds_df,
                target=target,
                train_days=selected_train_days,
                year=series_year,
                resample_freq=resample_freq,
            )

            if actual_df.empty or pred_df.empty:
                print(f"No prediction rows for target={target}, train_days={selected_train_days}.")
                continue

            fig = go.Figure()
            add_season_bands(fig, station["seasons"], station["season_order"])

            fig.add_trace(
                go.Scatter(
                    x=actual_df["Time"],
                    y=actual_df["Actual"],
                    mode="lines",
                    name="Actual",
                    line=dict(color="black", width=2.0),
                )
            )

            for model_name in MODEL_ORDER:
                model_frame = pred_df[pred_df["Model"] == model_name].copy()
                if model_frame.empty:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=model_frame["Time"],
                        y=model_frame["Predicted"],
                        mode="lines",
                        name=model_name,
                        line=dict(
                            color=MODEL_COLORS.get(model_name, "#666666"),
                            width=1.8,
                            dash=LINE_STYLES.get(model_name, "solid"),
                        ),
                    )
                )

            for annotation in season_metric_annotations(pred_df, actual_df, station["seasons"], station["season_order"]):
                fig.add_annotation(
                    x=annotation["x"],
                    y=0.02,
                    xref="x",
                    yref="paper",
                    text=annotation["text"],
                    showarrow=False,
                    font=dict(size=8, color="#333"),
                    bgcolor="rgba(255,255,255,0.80)",
                )

            title_target = TARGET_TITLES.get(target, target)
            subtitle_parts = [f"train={selected_train_days} d"]
            if station["latitude"] is not None and station["longitude"] is not None:
                subtitle_parts.append(f"{station['latitude']:.3f}N, {station['longitude']:.3f}E")
            if resample_freq:
                subtitle_parts.append(f"resample={resample_freq}")

            fig.update_layout(
                template=PLOTLY_TEMPLATE,
                height=540,
                hovermode="x unified",
                legend=dict(orientation="h", y=-0.20),
                xaxis_title="Date",
                yaxis_title=title_target,
                title=(
                    f"Forecast vs actual: {title_target} | {station['station_name']} ({station['station_code']})"
                    f"<br><sup>{' | '.join(subtitle_parts)}</sup>"
                ),
            )
            fig.show()
        """
    ),
    md_cell(
        """
        ## Graph 5. SHAP Importance Trends

        Recreates the seasonal SHAP time-series graph. By default it plots the top `shap_top_features`
        features for readability; set that parameter to `None` to plot all available features.
        """
    ),
    code_cell(
        """
        for target in sorted(shap_df["Target"].dropna().unique()):
            feature_list = top_shap_features(
                shap_df,
                target=target,
                train_days=selected_train_days,
                top_n=shap_top_features,
            )
            target_shap_df = shap_df[
                (shap_df["Target"] == target)
                & (shap_df["TrainDays"] == selected_train_days)
                & (shap_df["Feature"].isin(feature_list))
            ].copy()

            if target_shap_df.empty:
                print(f"No SHAP rows for target={target}, train_days={selected_train_days}.")
                continue

            fig = go.Figure()
            add_season_bands(fig, station["seasons"], station["season_order"])

            shown = set()
            for (model_name, feature_name), group in target_shap_df.groupby(["Model", "Feature"]):
                group = group.sort_values("Date")
                label = f"{feature_name} [{model_name}]"
                show_legend = label not in shown
                shown.add(label)

                fig.add_trace(
                    go.Scatter(
                        x=group["Date"],
                        y=group["Pct_s"],
                        mode="lines",
                        name=label,
                        legendgroup=f"{feature_name}_{model_name}",
                        showlegend=show_legend,
                        line=dict(
                            color=get_feature_color(feature_name),
                            width=LINE_WIDTHS.get(model_name, 2.0),
                            dash=LINE_STYLES.get(model_name, "solid"),
                        ),
                        hovertemplate=(
                            "<b>" + feature_name + "</b> [" + model_name + "]: %{y:.1f}%<extra></extra>"
                        ),
                    )
                )

            fig.update_layout(
                template=PLOTLY_TEMPLATE,
                height=560,
                hovermode="x unified",
                margin=dict(t=90, b=60, l=70, r=260),
                legend=dict(
                    orientation="v",
                    x=1.01,
                    y=1,
                    xanchor="left",
                    yanchor="top",
                    font=dict(size=10),
                    title=dict(text="Feature [Model]"),
                ),
                xaxis=dict(title="Date"),
                yaxis=dict(title="Share of |SHAP|, %"),
                title=(
                    f"SHAP feature importance: {target} | {station['station_name']} ({station['station_code']})"
                    f"<br><sup>train={selected_train_days} d | top_features={shap_top_features}</sup>"
                ),
            )
            fig.show()
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1, ensure_ascii=True) + "\n", encoding="utf-8")
print(f"Wrote {NOTEBOOK_PATH}")
