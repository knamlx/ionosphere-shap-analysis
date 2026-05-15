from datetime import datetime
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import ionosphere_pipeline as pipeline
import ml_pipeline_runner as runner


def make_hourly_source_frame(start: str = "2025-01-01", periods: int = 120, freq: str = "1h") -> pd.DataFrame:
    time = pd.date_range(start, periods=periods, freq=freq, tz="UTC")
    phase = np.arange(periods, dtype=float)
    tec = 10.0 + phase * 0.05
    return pd.DataFrame(
        {
            "Time": time,
            "CS": np.full(periods, 100.0),
            "foF2": 4.0 + 0.02 * phase + 0.2 * np.sin(phase / 6.0),
            "MUFD": 14.0 + 0.03 * phase + 0.3 * np.cos(phase / 5.0),
            "TEC": tec,
            "foEs": 2.0 + 0.1 * np.sin(phase / 8.0),
            "hmF2": 240.0 + 0.5 * phase,
            "foE": 3.0 + 0.05 * np.cos(phase / 7.0),
            "B0": 90.0 + 0.1 * phase,
        }
    )


def write_minimal_dataset(path, station_code: str = "TS001", station_name: str = "TEST SITE") -> None:
    path.write_text(
        "\n".join(
            [
                "# Global Ionospheric Radio Observatory",
                f"# Location: GEO 35.000N 33.000E, URSI-Code {station_code} {station_name}",
                "#Time CS foF2 foF2_QD",
                "2025-01-01T00:00:00.000Z 100 5.0 1",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_mufd_dataset(path, rows: list[str], station_code: str = "TS001", station_name: str = "TEST SITE") -> None:
    path.write_text(
        "\n".join(
            [
                "# Global Ionospheric Radio Observatory",
                f"# Location: GEO 35.000N 33.000E, URSI-Code {station_code} {station_name}",
                "#Time CS foF2 foF2_QD MUFD MUFD_QD hmF2 hmF2_QD",
                *rows,
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_build_hourly_frame_aligns_future_target_without_wrapping() -> None:
    config = pipeline.PipelineConfig(forecast_h=24, targets=("foF2",))
    df_h = pipeline.build_hourly_frame(make_hourly_source_frame(periods=72), config=config)

    assert np.isclose(
        df_h.loc[0, "foF2_target"],
        df_h.loc[24, "foF2"],
    ), "The forecast target must point exactly forecast_h hours into the future."
    assert df_h["foF2_target"].tail(config.forecast_h).isna().all(), (
        "The tail must stay NaN instead of silently wrapping targets across the dataset boundary."
    )


def test_build_hourly_frame_infers_dataset_step_from_source_data() -> None:
    config = pipeline.PipelineConfig(forecast_h=2, targets=("foF2",))
    df_h = pipeline.build_hourly_frame(make_hourly_source_frame(periods=12, freq="30min"), config=config)

    assert df_h.attrs["dataset_step"] == pd.Timedelta(minutes=30)
    assert (df_h["Time"].diff().dropna() == pd.Timedelta(minutes=30)).all()
    assert np.isclose(
        df_h.loc[0, "foF2_target"],
        df_h.loc[4, "foF2"],
    ), "A 2-hour horizon on a 30-minute dataset must shift by 4 rows."


def test_infer_dataset_step_rejects_pathological_subminute_artifacts() -> None:
    time = [
        pd.Timestamp("2025-01-01T00:00:00Z"),
        pd.Timestamp("2025-01-01T00:00:00.300Z"),
        pd.Timestamp("2025-01-01T00:05:00Z"),
        pd.Timestamp("2025-01-01T00:05:00.300Z"),
        pd.Timestamp("2025-01-01T00:10:00Z"),
        pd.Timestamp("2025-01-01T00:10:00.300Z"),
        pd.Timestamp("2025-01-01T00:15:00Z"),
        pd.Timestamp("2025-01-01T00:15:00.300Z"),
    ]
    df = pd.DataFrame({"Time": time})

    assert pipeline.infer_dataset_step(df) == pd.Timedelta(minutes=5)


def test_build_hourly_frame_uses_configured_dataset_step_override() -> None:
    config = pipeline.PipelineConfig(dataset_step="1h", forecast_h=2, targets=("foF2",))
    df_h = pipeline.build_hourly_frame(make_hourly_source_frame(periods=12, freq="30min"), config=config)

    assert df_h.attrs["dataset_step"] == pd.Timedelta(hours=1)
    assert (df_h["Time"].diff().dropna() == pd.Timedelta(hours=1)).all()
    assert np.isclose(
        df_h.loc[0, "foF2_target"],
        df_h.loc[2, "foF2"],
    ), "A configured 1-hour step must keep a 2-hour horizon at 2 rows after resampling."


def test_build_hourly_frame_applies_causal_savgol_to_target_columns() -> None:
    base = make_hourly_source_frame(periods=168)
    shifted = base.copy()
    shifted.loc[130:132, "foF2"] += 50.0
    config = pipeline.PipelineConfig(
        forecast_h=24,
        window_list=(2,),
        plot_train_days=2,
        savgol_target_columns=("foF2",),
        savgol_polyorder=2,
        targets=("foF2",),
        seasons={"Eval Day": (datetime(2025, 1, 5), datetime(2025, 1, 5))},
    )

    df_base = pipeline.build_hourly_frame(base, config=config)
    df_shifted = pipeline.build_hourly_frame(shifted, config=config)

    assert "foF2_savgol" in df_base.columns
    savgol_diag = pd.DataFrame(df_base.attrs["savgol_diagnostics_records"])
    assert bool(savgol_diag.loc[0, "Applied"])
    savgol_metadata = df_base.attrs["savgol_metadata"]
    assert bool(savgol_metadata["enabled"])
    assert savgol_metadata["requested_target_columns"] == ["foF2"]
    assert savgol_metadata["target_label_source"] == "filtered"
    assert savgol_metadata["columns"][0]["column"] == "foF2"
    assert savgol_metadata["columns"][0]["output_column"] == "foF2_savgol"
    assert np.allclose(
        df_base.loc[:129, "foF2_savgol"],
        df_shifted.loc[:129, "foF2_savgol"],
        equal_nan=True,
    ), "The Savitzky-Golay target filter must be causal and ignore future spikes."
    assert np.isclose(df_base.loc[0, "foF2_target"], df_base.loc[24, "foF2_savgol"])


def test_build_hourly_frame_reports_savgol_final_parameters(capsys) -> None:
    config = pipeline.PipelineConfig(
        forecast_h=24,
        savgol_target_columns=("foF2",),
        savgol_polyorder=2,
        targets=("foF2",),
        verbose=1,
    )

    pipeline.build_hourly_frame(make_hourly_source_frame(periods=168), config=config)
    captured = capsys.readouterr()

    assert "Savitzky-Golay final settings" in captured.out
    assert "Savitzky-Golay final parameters [foF2 -> foF2_savgol]" in captured.out
    assert "polyorder=2" in captured.out


def test_load_giro_dataset_keeps_valid_high_mufd_values() -> None:
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "HIGH_MUFD_2025.txt"
        write_mufd_dataset(
            path,
            rows=[
                "2025-03-20T07:00:00.000Z 100 10.000 1 35.000 1 250.000 1",
                "2025-03-20T07:05:00.000Z 100 10.200 1 36.500 1 255.000 1",
            ],
        )
        df = pipeline.load_giro_dataset({2025: path}, min_cs=0, verbose=0)

    assert np.isclose(df.loc[0, "MUFD"], 35.0)
    assert np.isclose(df.loc[1, "MUFD"], 36.5)


def test_restore_mufd_long_gaps_rebuilds_from_fof2() -> None:
    time = pd.date_range("2025-03-20T00:00:00Z", periods=96, freq="5min")
    fo_f2 = 8.0 + np.linspace(0.0, 2.0, len(time))
    mufd = fo_f2 * 3.4
    mufd[12:73] = np.nan
    df = pd.DataFrame(
        {
            "Time": time,
            "foF2": fo_f2,
            "MUFD": mufd,
            "hmF2": np.full(len(time), 260.0),
        }
    )

    restored, summary = pipeline.restore_mufd_long_gaps(df)

    assert not summary.empty
    assert int(summary.loc[0, "RestoredRows"]) == 61
    assert np.allclose(restored.loc[12:72, "MUFD"], restored.loc[12:72, "foF2"] * 3.4)


def test_build_dataset_frame_can_choose_filtered_or_raw_target_labels_and_state_features() -> None:
    base = make_hourly_source_frame(periods=168)
    filtered_config = pipeline.PipelineConfig(
        forecast_h=24,
        window_list=(2,),
        plot_train_days=2,
        savgol_target_columns=("foF2", "MUFD"),
        savgol_polyorder=2,
        use_filtered_target_labels=True,
        targets=("foF2", "MUFD"),
        seasons={"Eval Day": (datetime(2025, 1, 5), datetime(2025, 1, 5))},
    )
    raw_config = pipeline.PipelineConfig(
        forecast_h=24,
        window_list=(2,),
        plot_train_days=2,
        savgol_target_columns=("foF2", "MUFD"),
        savgol_polyorder=2,
        use_filtered_target_labels=False,
        targets=("foF2", "MUFD"),
        seasons={"Eval Day": (datetime(2025, 1, 5), datetime(2025, 1, 5))},
    )

    df_filtered = pipeline.build_dataset_frame(base, config=filtered_config)
    df_raw_labels = pipeline.build_dataset_frame(base, config=raw_config)

    assert df_filtered.attrs["target_label_source"] == "filtered"
    assert df_raw_labels.attrs["target_label_source"] == "raw"
    assert np.isclose(df_filtered.loc[0, "foF2_target"], df_filtered.loc[24, "foF2_savgol"])
    assert np.isclose(df_raw_labels.loc[0, "foF2_target"], df_raw_labels.loc[24, "foF2"])
    assert np.isclose(df_filtered.loc[0, "foF2_state"], df_filtered.loc[0, "foF2_savgol"])
    assert np.isclose(df_filtered.loc[0, "MUFD_state"], df_filtered.loc[0, "MUFD_savgol"])
    assert np.isclose(df_raw_labels.loc[0, "foF2_state"], df_raw_labels.loc[0, "foF2"])
    assert np.isclose(df_raw_labels.loc[0, "MUFD_state"], df_raw_labels.loc[0, "MUFD"])
    assert df_filtered.attrs["target_series_sources"] == {"foF2": "filtered", "MUFD": "filtered"}
    assert df_raw_labels.attrs["target_series_sources"] == {"foF2": "raw", "MUFD": "raw"}
    feature_pool = pipeline.get_feature_pool(df_filtered)
    assert "foF2_state" in feature_pool
    assert "MUFD_state" in feature_pool


def test_build_hourly_frame_verbose_reports_savgol_parameters(capsys) -> None:
    config = pipeline.PipelineConfig(
        forecast_h=24,
        window_list=(2,),
        plot_train_days=2,
        verbose=1,
        savgol_target_columns=("foF2",),
        savgol_polyorder=2,
        targets=("foF2",),
        seasons={"Eval Day": (datetime(2025, 1, 5), datetime(2025, 1, 5))},
    )

    pipeline.build_hourly_frame(make_hourly_source_frame(periods=168), config=config)
    captured = capsys.readouterr()

    assert "Savitzky-Golay final settings" in captured.out
    assert "Savitzky-Golay final parameters [foF2 -> foF2_savgol]" in captured.out
    assert "polyorder=2" in captured.out
    assert "window=" in captured.out
    assert "Target-derived ML feature sources:" in captured.out


def test_build_daily_frame_can_approximate_sparse_input_data() -> None:
    df = pd.DataFrame(
        {
            "Time": pd.to_datetime(
                [
                    "2025-01-01T00:00:00Z",
                    "2025-01-03T00:00:00Z",
                ],
                utc=True,
            ),
            "foF2": [10.0, 20.0],
            "MUFD": [30.0, 50.0],
            "foE": [1.0, 3.0],
            "hmF2": [200.0, 260.0],
        }
    )

    plain_daily = pipeline.build_daily_frame(df)
    approx_daily = pipeline.build_daily_frame(
        df,
        config=pipeline.PipelineConfig(dataset_step="1h", daily_approximate_input=True),
    )

    middle_day = pd.Timestamp("2025-01-02T00:00:00Z")
    plain_middle = plain_daily.loc[plain_daily["Time"] == middle_day, "foF2"].iloc[0]
    approx_middle = approx_daily.loc[approx_daily["Time"] == middle_day, "foF2"].iloc[0]

    assert np.isnan(plain_middle)
    assert np.isclose(approx_middle, 17.395833333333332, atol=1e-9)


def test_get_feature_pool_rejects_generic_future_columns() -> None:
    df = pd.DataFrame(
        {
            "feature_ok": [1.0, 2.0],
            "spoiler_target": [3.0, 4.0],
            "shortcut_pred": [5.0, 6.0],
            "foF2_savgol": [7.5, 8.5],
            "foF2": [7.0, 8.0],
            "hour": [0.0, 1.0],
        }
    )

    assert pipeline.get_feature_pool(df) == [
        "feature_ok"
    ], "Future-derived columns must never slip into the learning feature pool."


def test_list_available_stations_respects_verbose_flag(tmp_path, capsys) -> None:
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    write_minimal_dataset(datasets_dir / "TESTSITE_2025.txt")

    stations = pipeline.list_available_stations(base_dir=tmp_path, datasets_dir="datasets", verbose=0)
    quiet_output = capsys.readouterr()

    assert quiet_output.out == ""
    assert stations.loc[0, "station_code"] == "TS001"

    pipeline.list_available_stations(base_dir=tmp_path, datasets_dir="datasets", verbose=1)
    verbose_output = capsys.readouterr()

    assert "Found dataset file:" in verbose_output.out
    assert "TESTSITE_2025.txt" in verbose_output.out


def test_parse_f107_adjusted_flux_records_reads_latest_value_per_day() -> None:
    raw_table = "\n".join(
        [
            "fluxdate    fluxtime    fluxjulian    fluxcarrington  fluxobsflux  fluxadjflux  fluxursi",
            "----------  ----------  ------------  --------------  -----------  -----------  ----------",
            "20260430    170000      2461161.197   2310.55         0142.4       0144.5       0130.1",
            "20260430    230000      2461161.447   2310.56         0142.0       0144.1       0129.7",
            "20260501    170000      2461162.197   2310.59         0145.4       0147.6       0132.8",
        ]
    )

    records = pipeline.parse_f107_adjusted_flux_records(raw_table)

    assert records[datetime(2026, 4, 30).date()] == 144.1
    assert records[datetime(2026, 5, 1).date()] == 147.6


def test_resolve_iri_config_from_metadata_enriches_station_values(monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "fetch_station_altitude_km", lambda latitude, longitude: 0.163)

    resolved = pipeline.resolve_iri_config_from_metadata(
        {"latitude": 50.85, "longitude": 6.02},
        iri_config=pipeline.IRIConfig(),
    )

    assert resolved.lat_station == 50.85
    assert resolved.lon_station == 6.02
    assert resolved.alt_km == 0.163
    assert resolved.f107_val == pipeline.IRIConfig().f107_val


def test_resolve_iri_config_from_metadata_keeps_fallbacks_when_lookups_fail(monkeypatch) -> None:
    monkeypatch.setattr(
        pipeline,
        "fetch_station_altitude_km",
        lambda latitude, longitude: (_ for _ in ()).throw(RuntimeError("elevation down")),
    )
    original = pipeline.IRIConfig(lat_station=10.0, lon_station=20.0, alt_km=0.3, f107_val=155.0)
    resolved = pipeline.resolve_iri_config_from_metadata(
        {"latitude": 50.85, "longitude": 6.02},
        iri_config=original,
    )

    assert resolved.lat_station == 50.85
    assert resolved.lon_station == 6.02
    assert resolved.alt_km == original.alt_km
    assert resolved.f107_val == original.f107_val


def test_fetch_station_altitude_km_uses_nearby_land_when_exact_point_is_sea_level(monkeypatch) -> None:
    pipeline.fetch_station_altitude_km.cache_clear()
    monkeypatch.setattr(
        pipeline,
        "fetch_remote_text",
        lambda url, **kwargs: json.dumps(
            {
                "results": [
                    {"latitude": 38.0, "longitude": 23.5, "elevation": 3.0},
                    {"latitude": 38.05, "longitude": 23.5, "elevation": 84.0},
                    {"latitude": 37.95, "longitude": 23.5, "elevation": 18.0},
                    {"latitude": 38.0, "longitude": 23.55, "elevation": 0.0},
                    {"latitude": 38.0, "longitude": 23.45, "elevation": 0.0},
                    {"latitude": 38.05, "longitude": 23.55, "elevation": 11.0},
                    {"latitude": 38.05, "longitude": 23.45, "elevation": 0.0},
                    {"latitude": 37.95, "longitude": 23.55, "elevation": 0.0},
                    {"latitude": 37.95, "longitude": 23.45, "elevation": 0.0},
                ]
            }
        ),
    )

    assert pipeline.fetch_station_altitude_km(38.0, 23.5) == 0.084
    pipeline.fetch_station_altitude_km.cache_clear()


def test_resolve_f107_adjusted_flux_for_date_prefers_exact_day_then_month_then_fallback(monkeypatch) -> None:
    monkeypatch.setattr(
        pipeline,
        "fetch_f107_adjusted_flux_records",
        lambda: {
            datetime(2025, 3, 1).date(): 120.0,
            datetime(2025, 3, 20).date(): 130.0,
            datetime(2025, 4, 2).date(): 140.0,
        },
    )

    assert pipeline.resolve_f107_adjusted_flux_for_date(datetime(2025, 3, 1).date(), fallback=150.0) == 120.0
    assert pipeline.resolve_f107_adjusted_flux_for_date(datetime(2025, 3, 15).date(), fallback=150.0) == 130.0
    assert pipeline.resolve_f107_adjusted_flux_for_date(datetime(2025, 2, 1).date(), fallback=150.0) == 150.0


def test_resolve_f107_adjusted_flux_for_date_uses_fallback_when_source_fails(monkeypatch) -> None:
    monkeypatch.setattr(
        pipeline,
        "fetch_f107_adjusted_flux_records",
        lambda: (_ for _ in ()).throw(RuntimeError("space weather down")),
    )

    assert pipeline.resolve_f107_adjusted_flux_for_date(datetime(2025, 3, 15).date(), fallback=155.0) == 155.0


def test_split_walk_forward_window_discards_rows_with_future_labels() -> None:
    config = pipeline.PipelineConfig(forecast_h=24, targets=("foF2",))
    df_h = pipeline.build_hourly_frame(make_hourly_source_frame(periods=120), config=config)

    train_frame, test_frame, bounds = pipeline.split_walk_forward_window(
        df_h=df_h,
        current_day=datetime(2025, 1, 4),
        train_days=2,
        forecast_h=config.forecast_h,
    )

    assert len(train_frame) == 24, "The leak-free split must drop the last forecast_h training hours."
    assert len(test_frame) == 24, "A daily evaluation window must still keep the full test day."
    assert (
        train_frame["Time"] + pd.Timedelta(hours=config.forecast_h) <= bounds["train_end"]
    ).all(), "No training label may require observations from after the training cutoff."
    assert (
        train_frame["Time"] + pd.Timedelta(hours=config.forecast_h) < bounds["test_start"]
    ).all(), "No training label may land inside the future evaluation window."


def test_split_walk_forward_window_respects_inferred_half_hour_step() -> None:
    config = pipeline.PipelineConfig(forecast_h=24, targets=("foF2",))
    df_h = pipeline.build_hourly_frame(make_hourly_source_frame(periods=240, freq="30min"), config=config)

    train_frame, test_frame, bounds = pipeline.split_walk_forward_window(
        df_h=df_h,
        current_day=datetime(2025, 1, 4),
        train_days=2,
        forecast_h=config.forecast_h,
    )

    assert len(train_frame) == 48
    assert len(test_frame) == 48
    assert bounds["train_end"] == pd.Timestamp("2025-01-03 23:30:00+0000", tz="UTC")
    assert bounds["test_end"] == pd.Timestamp("2025-01-04 23:30:00+0000", tz="UTC")


def test_walk_forward_models_skip_windows_that_only_work_via_label_leakage(monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "make_models", lambda config=None: {"TinyLinear": LinearRegression()})
    config = pipeline.PipelineConfig(
        forecast_h=24,
        window_list=(1,),
        plot_train_days=1,
        targets=("foF2",),
        min_feature_coverage=0.0,
        min_train_rows=1,
        min_train_target_rows=1,
        min_eval_rows=1,
        seasons={"Leak Day": (datetime(2025, 1, 3), datetime(2025, 1, 3))},
    )
    df_h = pipeline.build_hourly_frame(make_hourly_source_frame(periods=96), config=config)

    metrics_df, fi_df, preds_df = pipeline.run_walk_forward_models(df_h, config=config)

    assert metrics_df.empty, "A window with only future labels must be rejected instead of scored."
    assert fi_df.empty, "Feature importance from a leaked split is invalid and must not be emitted."
    assert preds_df.empty, "Predictions from a leaked split must not be reported as if they were honest."


def test_walk_forward_models_verbose_reports_progress(monkeypatch, capsys) -> None:
    monkeypatch.setattr(pipeline, "make_models", lambda config=None: {"RandomForest": LinearRegression()})
    config = pipeline.PipelineConfig(
        forecast_h=24,
        window_list=(2,),
        plot_train_days=2,
        verbose=1,
        targets=("foF2",),
        min_feature_coverage=0.0,
        min_train_rows=1,
        min_train_target_rows=1,
        min_eval_rows=1,
        seasons={"Safe Day": (datetime(2025, 1, 4), datetime(2025, 1, 4))},
    )
    df_h = pipeline.build_hourly_frame(make_hourly_source_frame(periods=120), config=config)

    pipeline.run_walk_forward_models(df_h, config=config)
    captured = capsys.readouterr()

    assert "Walk-forward modeling started" in captured.out
    assert "Completed 2d walk-forward window" in captured.out
    assert "Walk-forward modeling finished" in captured.out


def test_walk_forward_models_still_score_when_safe_history_exists(monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "make_models", lambda config=None: {"RandomForest": LinearRegression()})
    config = pipeline.PipelineConfig(
        forecast_h=24,
        window_list=(2,),
        plot_train_days=2,
        verbose=0,
        targets=("foF2",),
        min_feature_coverage=0.0,
        min_train_rows=1,
        min_train_target_rows=1,
        min_eval_rows=1,
        seasons={"Safe Day": (datetime(2025, 1, 4), datetime(2025, 1, 4))},
    )
    df_h = pipeline.build_hourly_frame(make_hourly_source_frame(periods=120), config=config)

    metrics_df, fi_df, preds_df = pipeline.run_walk_forward_models(df_h, config=config)

    assert not metrics_df.empty, "The leakage guard must not suppress valid walk-forward experiments."
    assert set(metrics_df["Model"]) == {"RandomForest"}
    assert (metrics_df["N_train"] == 24).all(), "Only the leak-free portion of history should remain trainable."
    assert not fi_df.empty, "Valid training windows should still produce explainable feature rankings."
    assert not preds_df.empty, "Valid training windows should still produce evaluation predictions."


def test_run_walk_forward_models_records_predictions_for_all_windows_and_models(monkeypatch) -> None:
    monkeypatch.setattr(
        pipeline,
        "make_models",
        lambda config=None: {
            "RandomForest": LinearRegression(),
            "ElasticNet": LinearRegression(),
        },
    )
    config = pipeline.PipelineConfig(
        forecast_h=24,
        window_list=(2, 3),
        plot_train_days=2,
        targets=("foF2",),
        min_feature_coverage=0.0,
        min_train_rows=1,
        min_train_target_rows=1,
        min_eval_rows=1,
        seasons={"Safe Day": (datetime(2025, 1, 5), datetime(2025, 1, 5))},
    )
    df_h = pipeline.build_hourly_frame(make_hourly_source_frame(periods=168), config=config)

    _, _, preds_df = pipeline.run_walk_forward_models(df_h, config=config)

    assert not preds_df.empty
    assert set(preds_df["TrainDays"]) == {2, 3}
    assert set(preds_df["Model"]) == {"RandomForest", "ElasticNet"}
    assert (preds_df["TargetTime"] - preds_df["Time"] == pd.Timedelta(hours=config.forecast_h)).all()


class FakeTrial:
    def __init__(self, number: int) -> None:
        self.number = number
        self.params: dict[str, object] = {}
        self.value = None
        self.state = "COMPLETE"


class FakeStudy:
    def __init__(self, direction: str) -> None:
        self.direction = direction
        self.trials: list[FakeTrial] = []
        self.best_params: dict[str, object] = {}
        self.best_value = np.inf if direction == "minimize" else -np.inf

    def optimize(self, objective, n_trials: int) -> None:
        for number in range(n_trials):
            trial = FakeTrial(number)
            trial.value = objective(trial)
            self.trials.append(trial)
            if self.direction == "minimize":
                is_better = trial.value < self.best_value
            else:
                is_better = trial.value > self.best_value
            if is_better:
                self.best_value = trial.value
                self.best_params = dict(trial.params)


class FakeOptunaModule:
    class logging:
        INFO = 20
        WARNING = 30
        last_level = None

        @classmethod
        def set_verbosity(cls, level):
            cls.last_level = level

    class samplers:
        class TPESampler:
            def __init__(self, seed: int) -> None:
                self.seed = seed

    @staticmethod
    def create_study(direction: str, sampler) -> FakeStudy:
        del sampler
        return FakeStudy(direction=direction)


def test_configure_optuna_logging_uses_info_only_for_verbose_2() -> None:
    FakeOptunaModule.logging.last_level = None
    pipeline._configure_optuna_logging(FakeOptunaModule, verbose=1)
    assert FakeOptunaModule.logging.last_level == FakeOptunaModule.logging.WARNING

    pipeline._configure_optuna_logging(FakeOptunaModule, verbose=2)
    assert FakeOptunaModule.logging.last_level == FakeOptunaModule.logging.INFO


def test_split_train_val_test_window_keeps_validation_labels_out_of_test_period() -> None:
    config = pipeline.PipelineConfig(forecast_h=24, targets=("foF2",))
    df_h = pipeline.build_hourly_frame(make_hourly_source_frame(periods=240), config=config)

    train_frame, val_frame, test_frame, bounds = pipeline.split_train_val_test_window(
        df_h=df_h,
        current_day=datetime(2025, 1, 7),
        train_days=4,
        forecast_h=config.forecast_h,
        val_days=1,
        test_h=24,
    )

    assert len(train_frame) == 72
    assert len(val_frame) == 24
    assert len(test_frame) == 24
    assert (
        val_frame["Time"] + pd.Timedelta(hours=config.forecast_h) < bounds["test_start"]
    ).all(), "Validation labels must finish before the final test period starts."
    assert (
        train_frame["Time"] + pd.Timedelta(hours=config.forecast_h) < bounds["val_start"]
    ).all(), "Training labels must finish before the validation feature period starts."
    assert bounds["val_label_end"] < bounds["test_start"]


def test_split_train_val_test_window_defaults_validation_size_to_test_size() -> None:
    config = pipeline.PipelineConfig(forecast_h=24, targets=("foF2",))
    df_h = pipeline.build_hourly_frame(make_hourly_source_frame(periods=240), config=config)

    _, val_frame, test_frame, _ = pipeline.split_train_val_test_window(
        df_h=df_h,
        current_day=datetime(2025, 1, 7),
        train_days=4,
        forecast_h=config.forecast_h,
        val_days=2,
        test_h=48,
    )

    assert len(val_frame) == len(test_frame) == 48


def test_run_optuna_walk_forward_models_uses_train_val_test_flow(monkeypatch) -> None:
    fit_sizes: list[int] = []

    class MeanOffsetRegressor:
        def __init__(self, offset: float) -> None:
            self.offset = offset

        def fit(self, x_train, y_train):
            fit_sizes.append(len(y_train))
            self.mean_ = float(np.mean(y_train))
            return self

        def predict(self, x_test):
            return np.full(len(x_test), self.mean_ + self.offset, dtype=float)

    def fake_sample_params(model_name: str, trial: FakeTrial) -> dict[str, object]:
        assert model_name == "RandomForest"
        trial.params = {"offset": float(trial.number)}
        return dict(trial.params)

    def fake_build_estimator(model_name: str, params: dict[str, object], random_state: int, model_n_jobs: int):
        del random_state, model_n_jobs
        assert model_name == "RandomForest"
        return MeanOffsetRegressor(offset=float(params["offset"]))

    monkeypatch.setattr(pipeline, "_load_optuna", lambda: FakeOptunaModule)
    monkeypatch.setattr(pipeline, "_sample_optuna_params", fake_sample_params)
    monkeypatch.setattr(pipeline, "_build_optuna_estimator", fake_build_estimator)

    config = pipeline.PipelineConfig(
        forecast_h=24,
        window_list=(4,),
        plot_train_days=4,
        targets=("foF2",),
        min_feature_coverage=0.0,
        min_train_rows=1,
        min_train_target_rows=1,
        min_eval_rows=1,
        seasons={"Optuna Day": (datetime(2025, 1, 7), datetime(2025, 1, 7))},
    )
    optuna_config = pipeline.OptunaConfig(
        n_trials=2,
        val_days=None,
        test_h=24,
        metric="MAE",
        models=("RandomForest",),
        random_state=42,
        train_final_on_train_val=True,
    )
    df_h = pipeline.build_hourly_frame(make_hourly_source_frame(periods=240), config=config)

    metrics_df, trials_df, preds_df = pipeline.run_optuna_walk_forward_models(
        df_h=df_h,
        config=config,
        optuna_config=optuna_config,
    )

    assert not metrics_df.empty
    assert not trials_df.empty
    assert not preds_df.empty
    assert fit_sizes[:2] == [72, 72], "Optuna trials must fit on the train split only."
    assert fit_sizes[-1] == 96, "The final model must refit on train+validation before touching the test split."
    assert (metrics_df["ValDays"] == 1).all()
    assert (metrics_df["N_train"] == 72).all()
    assert (metrics_df["N_val"] == 24).all()
    assert (metrics_df["N_final_train"] == 96).all()
    train_frame, val_frame, _, _ = pipeline.split_train_val_test_window(
        df_h=df_h,
        current_day=datetime(2025, 1, 7),
        train_days=4,
        forecast_h=config.forecast_h,
        val_days=1,
        test_h=24,
    )
    train_mean = float(train_frame["foF2_target"].dropna().mean())
    val_target = val_frame["foF2_target"].dropna()
    score_offset_0 = float(np.mean(np.abs(val_target - train_mean)))
    score_offset_1 = float(np.mean(np.abs(val_target - (train_mean + 1.0))))
    best_params = metrics_df["BestParams"].map(json.loads).iloc[0]
    expected_offset = 0.0 if score_offset_0 <= score_offset_1 else 1.0
    assert best_params == {"offset": expected_offset}


def test_run_optuna_walk_forward_models_collects_iteration_shap_records(monkeypatch) -> None:
    class MeanRegressor:
        def fit(self, x_train, y_train):
            self.mean_ = float(np.mean(y_train))
            return self

        def predict(self, x_test):
            return np.full(len(x_test), self.mean_, dtype=float)

    exported_frames: list[pd.DataFrame] = []
    exported_contexts: list[dict[str, object]] = []

    monkeypatch.setattr(pipeline, "_load_optuna", lambda: FakeOptunaModule)
    monkeypatch.setattr(pipeline, "_sample_optuna_params", lambda model_name, trial: {})
    monkeypatch.setattr(
        pipeline,
        "_build_optuna_estimator",
        lambda model_name, params, random_state, model_n_jobs: MeanRegressor(),
    )
    monkeypatch.setattr(
        pipeline,
        "compute_optuna_iteration_shap_records",
        lambda **kwargs: pd.DataFrame(
            [
                {
                    "TrainDays": kwargs["train_days"],
                    "ValDays": kwargs["val_days"],
                    "TestH": kwargs["test_h"],
                    "Date": pd.Timestamp(kwargs["current_day"]),
                    "Season": kwargs["season_name"],
                    "Target": kwargs["target"],
                    "Model": kwargs["model_name"],
                    "Feature": "TEC",
                    "MeanAbsShap": 1.0,
                    "Pct": 100.0,
                    "NBackground": 3,
                    "NExplain": 2,
                }
            ]
        ),
    )

    config = pipeline.PipelineConfig(
        forecast_h=24,
        window_list=(4,),
        plot_train_days=4,
        targets=("foF2",),
        min_feature_coverage=0.0,
        min_train_rows=1,
        min_train_target_rows=1,
        min_eval_rows=1,
        seasons={"Optuna Day": (datetime(2025, 1, 7), datetime(2025, 1, 7))},
    )
    optuna_config = pipeline.OptunaConfig(
        n_trials=1,
        val_days=1,
        test_h=24,
        metric="MAE",
        models=("RandomForest",),
        random_state=42,
        train_final_on_train_val=True,
    )
    df_h = pipeline.build_hourly_frame(make_hourly_source_frame(periods=240), config=config)
    shap_records: list[dict[str, object]] = []

    pipeline.run_optuna_walk_forward_models(
        df_h=df_h,
        config=config,
        optuna_config=optuna_config,
        shap_record_sink=shap_records,
        shap_iteration_exporter=lambda shap_df, ctx: (
            exported_frames.append(shap_df.copy()),
            exported_contexts.append(dict(ctx)),
        ),
    )

    assert len(shap_records) == 1
    assert len(exported_frames) == 1
    assert exported_frames[0].iloc[0]["Pct"] == 100.0
    assert exported_contexts[0]["train_days"] == 4
    assert exported_contexts[0]["target"] == "foF2"
    assert exported_contexts[0]["model_name"] == "RandomForest"


def test_compute_optuna_iteration_shap_records_uses_xgboost_native_contribs() -> None:
    class FakeBooster:
        def predict(self, dmatrix, pred_contribs=False):
            assert pred_contribs is True
            rows = dmatrix.num_row()
            return np.tile(np.array([[0.2, 0.3, 0.5, 1.0]], dtype=float), (rows, 1))

    class FakeXGBModel:
        def get_booster(self):
            return FakeBooster()

    explain_frame = pd.DataFrame(
        {
            "TEC": [1.0, 2.0, 3.0],
            "foE": [4.0, 5.0, 6.0],
            "hmF2": [7.0, 8.0, 9.0],
        }
    )
    records = pipeline.compute_optuna_iteration_shap_records(
        model=FakeXGBModel(),
        x_train=explain_frame,
        x_test=explain_frame,
        train_days=7,
        val_days=1,
        test_h=24,
        season_name="Winter",
        current_day=datetime(2025, 1, 7),
        target="foF2",
        model_name="XGBoost",
        config=pipeline.PipelineConfig(shap_background=3, shap_sample=3),
    )

    assert list(records["Feature"]) == ["TEC", "foE", "hmF2"]
    assert np.isclose(records["Pct"].sum(), 100.0)
    assert (records["NExplain"] == 3).all()
    assert (records["NBackground"] == 3).all()


def test_compute_optuna_iteration_shap_records_supports_elasticnet_pipeline() -> None:
    explain_frame = pd.DataFrame(
        {
            "TEC": np.linspace(1.0, 6.0, 6),
            "foE": np.linspace(2.0, 7.0, 6),
            "hmF2": np.linspace(3.0, 8.0, 6),
        }
    )
    y = np.linspace(10.0, 15.0, 6)
    model = pipeline.Pipeline(
        [
            ("scaler", pipeline.StandardScaler()),
            ("model", pipeline.ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000, random_state=42)),
        ]
    )
    model.fit(explain_frame, y)

    records = pipeline.compute_optuna_iteration_shap_records(
        model=model,
        x_train=explain_frame,
        x_test=explain_frame,
        train_days=7,
        val_days=1,
        test_h=24,
        season_name="Winter",
        current_day=datetime(2025, 1, 7),
        target="foF2",
        model_name="ElasticNet",
        config=pipeline.PipelineConfig(shap_background=6, shap_sample=6),
    )

    assert list(records["Feature"]) == ["TEC", "foE", "hmF2"]
    assert np.isclose(records["Pct"].sum(), 100.0)
    assert (records["NExplain"] == 6).all()
    assert (records["NBackground"] == 6).all()


def test_run_walk_forward_models_honors_ml_date_range(monkeypatch) -> None:
    monkeypatch.setattr(
        pipeline,
        "make_models",
        lambda config=None: {"RandomForest": LinearRegression()},
    )
    config = pipeline.PipelineConfig(
        forecast_h=24,
        window_list=(2,),
        plot_train_days=2,
        targets=("foF2",),
        min_feature_coverage=0.0,
        min_train_rows=1,
        min_train_target_rows=1,
        min_eval_rows=1,
        ml_date_start="2025-01-05",
        ml_date_end="2025-01-05",
        seasons={"Two Days": (datetime(2025, 1, 4), datetime(2025, 1, 5))},
    )
    df_h = pipeline.build_hourly_frame(make_hourly_source_frame(periods=144), config=config)

    metrics_df, _, preds_df = pipeline.run_walk_forward_models(df_h, config=config)

    assert not metrics_df.empty
    assert set(metrics_df["Date"].dt.strftime("%Y-%m-%d")) == {"2025-01-05"}
    assert set(preds_df["Time"].dt.strftime("%Y-%m-%d")) == {"2025-01-05"}


def test_run_walk_forward_models_reports_percent_progress(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        pipeline,
        "make_models",
        lambda config=None: {"RandomForest": LinearRegression()},
    )
    config = pipeline.PipelineConfig(
        forecast_h=24,
        window_list=(2,),
        plot_train_days=2,
        targets=("foF2",),
        min_feature_coverage=0.0,
        min_train_rows=1,
        min_train_target_rows=1,
        min_eval_rows=1,
        verbose=1,
        seasons={"Eval Day": (datetime(2025, 1, 5), datetime(2025, 1, 5))},
    )
    df_h = pipeline.build_hourly_frame(make_hourly_source_frame(periods=144), config=config)

    pipeline.run_walk_forward_models(df_h, config=config)
    captured = capsys.readouterr()

    assert "Walk-forward progress" in captured.out
    assert "100%" in captured.out
    assert "remaining=0" in captured.out


def test_run_optuna_walk_forward_models_honors_ml_date_range(monkeypatch) -> None:
    class MeanRegressor:
        def fit(self, x_train, y_train):
            self.mean_ = float(np.mean(y_train))
            return self

        def predict(self, x_test):
            return np.full(len(x_test), self.mean_, dtype=float)

    monkeypatch.setattr(pipeline, "_load_optuna", lambda: FakeOptunaModule)
    monkeypatch.setattr(
        pipeline,
        "_sample_optuna_params",
        lambda model_name, trial: {},
    )
    monkeypatch.setattr(
        pipeline,
        "_build_optuna_estimator",
        lambda model_name, params, random_state, model_n_jobs: MeanRegressor(),
    )

    config = pipeline.PipelineConfig(
        forecast_h=24,
        window_list=(4,),
        plot_train_days=4,
        targets=("foF2",),
        min_feature_coverage=0.0,
        min_train_rows=1,
        min_train_target_rows=1,
        min_eval_rows=1,
        ml_date_start="2025-01-08",
        ml_date_end="2025-01-08",
        seasons={"Two Days": (datetime(2025, 1, 7), datetime(2025, 1, 8))},
    )
    optuna_config = pipeline.OptunaConfig(
        n_trials=1,
        val_days=1,
        test_h=24,
        metric="MAE",
        models=("RandomForest",),
        random_state=42,
        train_final_on_train_val=True,
    )
    df_h = pipeline.build_hourly_frame(make_hourly_source_frame(periods=264), config=config)

    metrics_df, trials_df, preds_df = pipeline.run_optuna_walk_forward_models(
        df_h=df_h,
        config=config,
        optuna_config=optuna_config,
    )

    assert not metrics_df.empty
    assert set(metrics_df["Date"].dt.strftime("%Y-%m-%d")) == {"2025-01-08"}
    assert set(trials_df["Date"].dt.strftime("%Y-%m-%d")) == {"2025-01-08"}
    assert set(preds_df["Time"].dt.strftime("%Y-%m-%d")) == {"2025-01-08"}


def test_compute_physical_metrics_honors_ml_date_range() -> None:
    config = pipeline.PipelineConfig(
        forecast_h=24,
        plot_train_days=2,
        targets=("foF2",),
        min_eval_rows=1,
        ml_date_start="2025-01-05",
        ml_date_end="2025-01-05",
        seasons={"Two Days": (datetime(2025, 1, 4), datetime(2025, 1, 5))},
    )
    df_h = pipeline.build_hourly_frame(make_hourly_source_frame(periods=168), config=config)
    df_h["IRI_foF2_pred"] = df_h["foF2_target"]
    df_h["anal_foF2_pred"] = df_h["foF2_target"]

    phys_df = pipeline.compute_physical_metrics(df_h, config=config)

    assert not phys_df.empty
    assert set(phys_df["Date"].dt.strftime("%Y-%m-%d")) == {"2025-01-05"}


def test_build_excel_output_name_appends_sanitized_suffix_and_timestamp() -> None:
    assert (
        pipeline.build_excel_output_name(
            "metrics_all_models_2025_mac.xlsx",
            " station/run 01 ",
            excel_timestamp="260503_123456",
        )
        == "metrics_all_models_2025_mac_station_run_01_260503_123456.xlsx"
    )


def test_build_excel_output_name_supports_window_prefix() -> None:
    assert (
        pipeline.build_excel_output_name(
            "metrics_optuna_models_2025_mac.csv",
            "EA036",
            excel_timestamp="260504_120000",
            file_name_prefix="window 14d",
        )
        == "EA036_window_14d_metrics_optuna_models_2025_mac_260504_120000.csv"
    )


def test_resolve_station_excel_name_suffix_keeps_multi_station_names_unique() -> None:
    assert pipeline._resolve_station_excel_name_suffix(None, "ab12") == "AB12"
    assert pipeline._resolve_station_excel_name_suffix(None, "ab12", ensure_unique=True) == "AB12"
    assert pipeline._resolve_station_excel_name_suffix("experiment", "ab12", ensure_unique=True) == "experiment_AB12"
    assert pipeline._resolve_station_excel_name_suffix("experiment_{station_code}", "ab12", ensure_unique=True) == "experiment_AB12"
    assert pipeline._resolve_station_excel_name_suffix("experiment_AB12", "ab12", ensure_unique=True) == "experiment_AB12"


def test_resolve_results_output_dir_uses_station_subfolder(tmp_path) -> None:
    assert pipeline.resolve_results_output_dir(tmp_path, "ea036") == tmp_path / "results" / "EA036"
    assert pipeline.resolve_results_output_dir(tmp_path, " station 01 ") == tmp_path / "results" / "STATION_01"


def test_build_excel_output_name_moves_station_code_to_front() -> None:
    assert (
        pipeline.build_excel_output_name(
            "predicted_time_series_optuna_models_2025_mac.csv",
            "experiment_AT138",
            excel_timestamp="260504_223349",
        )
        == "AT138_predicted_time_series_optuna_models_2025_mac_experiment_260504_223349.csv"
    )


def test_resolve_window_parallel_settings_splits_cpu_across_windows(monkeypatch) -> None:
    monkeypatch.setattr(pipeline.os, "cpu_count", lambda: 32)
    workers, model_n_jobs = pipeline.resolve_window_parallel_settings(
        pipeline.PipelineConfig(window_list=(7, 14, 21, 30))
    )

    assert workers == 4
    assert model_n_jobs == 8


def test_runner_loads_optuna_json_config_and_dispatches(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "runner.json"
    config_path.write_text(
        json.dumps(
            {
                "pipeline": "optuna",
                "base_dir": ".",
                "datasets_dir": "datasets",
                "station_code": "AT138",
                "export_excel": False,
                "compute_shap": False,
                "verbose": 1,
                "pipeline_config": {
                    "window_list": [7, 21],
                    "targets": ["foF2"],
                    "seasons": {"Eval": ["2025-01-01T00:00:00", "2025-01-02T00:00:00"]},
                },
                "optuna_config": {
                    "models": ["ElasticNet"],
                    "n_trials": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_run_optuna_pipeline(**kwargs):
        captured.update(kwargs)
        return {
            "config": kwargs["config"],
            "optuna_config": kwargs["optuna_config"],
            "metrics_df": pd.DataFrame([{"Metric": 1.0}]),
            "trials_df": pd.DataFrame([{"Trial": 1}]),
            "preds_df": pd.DataFrame(),
            "summary_df": pd.DataFrame(),
            "shap_df": pd.DataFrame(),
            "station_metadata": {"station_code": "AT138"},
        }

    monkeypatch.setattr(runner, "run_optuna_pipeline", fake_run_optuna_pipeline)

    result = runner.run_from_config(config_path)

    assert isinstance(captured["config"], pipeline.PipelineConfig)
    assert captured["config"].window_list == (7, 21)
    assert captured["config"].targets == ("foF2",)
    assert "Eval" in captured["config"].seasons
    assert isinstance(captured["optuna_config"], pipeline.OptunaConfig)
    assert captured["optuna_config"].models == ("ElasticNet",)
    assert result["station_metadata"]["station_code"] == "AT138"


def test_runner_dispatches_standard_pipeline(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "runner_standard.json"
    config_path.write_text(
        json.dumps(
            {
                "pipeline": "standard",
                "export_excel": False,
                "compute_shap": False,
                "pipeline_config": {
                    "window_list": [7],
                    "targets": ["foF2", "MUFD"],
                },
            }
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_run_pipeline(**kwargs):
        captured.update(kwargs)
        return {
            "config": kwargs["config"],
            "metrics_df": pd.DataFrame(),
            "fi_df": pd.DataFrame(),
            "preds_df": pd.DataFrame(),
            "summary_df": pd.DataFrame(),
            "shap_df": pd.DataFrame(),
            "station_metadata": {},
        }

    monkeypatch.setattr(runner, "run_pipeline", fake_run_pipeline)

    runner.run_from_config(config_path)

    assert isinstance(captured["config"], pipeline.PipelineConfig)
    assert captured["config"].window_list == (7,)
    assert captured["config"].targets == ("foF2", "MUFD")
    assert isinstance(captured["iri_config"], pipeline.IRIConfig)


def test_runner_build_summary_includes_savgol_metadata_and_export_dir() -> None:
    summary = runner.build_summary(
        {
            "config": pipeline.PipelineConfig(),
            "station_metadata": {"station_code": "EA036"},
            "savgol_metadata": {"enabled": True, "requested_target_columns": ["foF2"]},
            "export_dir": "N:/shap_fo_MUF_pred/results/EA036",
            "metrics_df": pd.DataFrame([{"Metric": 1.0}]),
        }
    )

    assert summary["savgol_metadata"]["enabled"] is True
    assert summary["savgol_metadata"]["requested_target_columns"] == ["foF2"]
    assert summary["export_dir"] == "N:/shap_fo_MUF_pred/results/EA036"
    assert summary["metrics_df_rows"] == 1


def test_build_excel_timestamp_token_uses_short_second_precision() -> None:
    assert pipeline.build_excel_timestamp_token(datetime(2026, 5, 3, 14, 7, 9)) == "260503_140709"


def test_export_results_writes_prediction_csv_with_aim_columns(tmp_path, monkeypatch) -> None:
    timestamp = "260503_123456"
    monkeypatch.setattr(pipeline, "build_excel_timestamp_token", lambda exported_at=None: timestamp)
    config = pipeline.PipelineConfig(
        forecast_h=24,
        window_list=(7,),
        plot_train_days=7,
        savgol_target_columns=("foF2", "MUFD"),
        targets=("foF2", "MUFD"),
    )
    df_h = pipeline.build_hourly_frame(make_hourly_source_frame(periods=168), config=config)
    source_time = pd.Timestamp(df_h.loc[24, "Time"])
    target_time = source_time + pd.Timedelta(hours=config.forecast_h)
    preds_df = pd.DataFrame(
        [
            {
                "TrainDays": 7,
                "Date": "2025-01-02",
                "Season": "Winter",
                "Time": source_time,
                "TargetTime": target_time,
                "Target": "foF2",
                "Model": "RandomForest",
                "actual": float(df_h.loc[24, "foF2_target"]),
                "predicted": 5.5,
            }
        ]
    )

    pipeline.export_results(
        metrics_df=pd.DataFrame(),
        phys_df=pd.DataFrame(),
        summary_df=pd.DataFrame(),
        fi_df=pd.DataFrame(),
        output_dir=tmp_path,
        preds_df=preds_df,
        df_h=df_h,
        config=config,
    )

    all_windows_path = tmp_path / pipeline.build_excel_output_name(
        "predicted_time_series_all_models_2025_mac.csv",
        excel_timestamp=timestamp,
    )
    window_path = all_windows_path.with_name(f"{all_windows_path.stem}_window_7d{all_windows_path.suffix}")
    assert all_windows_path.exists()
    assert window_path.exists()

    export_df = pd.read_csv(window_path)
    assert {
        "TrainDays",
        "RunDate",
        "SourceTime",
        "TargetTime",
        "Target",
        "Model",
        "ActualLabel",
        "Predicted",
        "foF2_original",
        "foF2_filtered",
        "MUFD_original",
        "MUFD_filtered",
    }.issubset(export_df.columns)

    first_row = export_df.iloc[0]
    target_index = df_h.index[df_h["Time"] == target_time][0]
    assert np.isclose(first_row["foF2_original"], df_h.loc[target_index, "foF2"])
    assert np.isclose(first_row["MUFD_filtered"], df_h.loc[target_index, "MUFD_savgol"])


def test_export_results_supports_custom_excel_name_suffix(tmp_path, monkeypatch) -> None:
    suffix = " station/run 02 "
    timestamp = "260503_123456"
    monkeypatch.setattr(pipeline, "build_excel_timestamp_token", lambda exported_at=None: timestamp)

    pipeline.export_results(
        metrics_df=pd.DataFrame([{"Metric": 1.0}]),
        phys_df=pd.DataFrame(),
        summary_df=pd.DataFrame([{"Summary": 1.0}]),
        fi_df=pd.DataFrame([{"TrainDays": 7, "Season": "Winter", "Target": "foF2", "Model": "RF", "Feature": "TEC", "Importance": 1.0}]),
        output_dir=tmp_path,
        preds_df=pd.DataFrame(),
        df_h=pd.DataFrame(),
        excel_name_suffix=suffix,
    )

    assert (
        tmp_path
        / pipeline.build_partition_output_name(
            "metrics_all_models_2025_mac.csv",
            "summary",
            suffix,
            excel_timestamp=timestamp,
        )
    ).exists()
    assert (
        tmp_path
        / pipeline.build_partition_output_name(
            "feature_importance_2025_mac.csv",
            "daily_fi",
            suffix,
            excel_timestamp=timestamp,
        )
    ).exists()
    assert (
        tmp_path
        / pipeline.build_excel_output_name(
            "predicted_time_series_all_models_2025_mac.csv",
            suffix,
            excel_timestamp=timestamp,
        )
    ).exists()


def test_export_optuna_results_supports_custom_excel_name_suffix(tmp_path, monkeypatch) -> None:
    suffix = " station/run 03 "
    timestamp = "260503_123456"
    monkeypatch.setattr(pipeline, "build_excel_timestamp_token", lambda exported_at=None: timestamp)
    config = pipeline.PipelineConfig(
        forecast_h=24,
        window_list=(7,),
        plot_train_days=7,
        savgol_target_columns=("foF2", "MUFD"),
        targets=("foF2", "MUFD"),
    )
    df_h = pipeline.build_hourly_frame(make_hourly_source_frame(periods=168), config=config)
    df_h["IRI_foF2_pred"] = df_h["foF2_target"]
    df_h["anal_foF2_pred"] = df_h["foF2_target"]
    source_time = pd.Timestamp(df_h.loc[24, "Time"])
    preds_df = pd.DataFrame(
        [
            {
                "TrainDays": 7,
                "ValDays": 1,
                "TestH": 24,
                "Date": "2025-01-02",
                "Season": "Winter",
                "Time": source_time,
                "TargetTime": source_time + pd.Timedelta(hours=config.forecast_h),
                "Target": "foF2",
                "Model": "RandomForest",
                "actual": float(df_h.loc[24, "foF2_target"]),
                "predicted": 5.5,
            }
        ]
    )

    pipeline.export_optuna_results(
        metrics_df=pd.DataFrame([{"Metric": 1.0}]),
        summary_df=pd.DataFrame([{"Summary": 1.0}]),
        trials_df=pd.DataFrame([{"Trial": 1}]),
        output_dir=tmp_path,
        phys_df=pd.DataFrame([{"Model": "IRI", "R2": 0.5, "MAE": 1.0}]),
        preds_df=preds_df,
        df_h=df_h,
        config=config,
        excel_name_suffix=suffix,
        shap_df=pd.DataFrame([{"TrainDays": 7, "Feature": "TEC"}]),
    )

    assert (
        tmp_path
        / pipeline.build_partition_output_name(
            "metrics_optuna_models_2025_mac.csv",
            "summary",
            suffix,
            excel_timestamp=timestamp,
        )
    ).exists()
    assert (
        tmp_path
        / pipeline.build_partition_output_name(
            "metrics_optuna_models_2025_mac.csv",
            "phys_daily",
            suffix,
            excel_timestamp=timestamp,
        )
    ).exists()
    assert (
        tmp_path
        / pipeline.build_excel_output_name(
            "predicted_time_series_optuna_models_2025_mac.csv",
            suffix,
            excel_timestamp=timestamp,
        )
    ).exists()
    assert (
        tmp_path
        / pipeline.build_excel_output_name(
            "shap_optuna_models_2025_mac.csv",
            suffix,
            excel_timestamp=timestamp,
        )
    ).exists()

    prediction_export_df = pd.read_csv(
        tmp_path
        / pipeline.build_excel_output_name(
            "predicted_time_series_optuna_models_2025_mac.csv",
            suffix,
            excel_timestamp=timestamp,
        )
    )
    assert {"RandomForest", "IRI", "Аналит."}.issubset(set(prediction_export_df["Model"]))


def test_build_optuna_summary_table_includes_physical_baselines() -> None:
    config = pipeline.PipelineConfig(plot_train_days=21, targets=("foF2",))
    metrics_df = pd.DataFrame(
        [
            {
                "TrainDays": 21,
                "ValDays": 1,
                "TestH": 24,
                "Season": "Winter",
                "Target": "foF2",
                "Model": "XGBoost",
                "R2": 0.8,
                "MAE": 1.1,
                "BestValScore": 0.9,
            }
        ]
    )
    phys_df = pd.DataFrame(
        [
            {
                "TrainDays": 21,
                "Season": "Winter",
                "Target": "foF2",
                "Model": "IRI",
                "R2": 0.5,
                "MAE": 1.6,
            },
            {
                "TrainDays": 21,
                "Season": "Winter",
                "Target": "foF2",
                "Model": "Analytic",
                "R2": 0.4,
                "MAE": 1.8,
            },
        ]
    )

    summary_df = pipeline.build_optuna_summary_table(metrics_df, phys_df=phys_df, config=config)

    assert set(summary_df["Model"]) == {"XGBoost", "IRI", "Analytic"}
    assert summary_df.loc[summary_df["Model"] == "IRI", "BestValScore"].isna().all()


def test_build_metric_comparison_table_supports_month_aggregation() -> None:
    metrics_df = pd.DataFrame(
        [
            {"TrainDays": 21, "Season": "Winter", "Date": "2025-01-15", "Target": "foF2", "Model": "RandomForest", "R2": 0.8, "MAE": 1.0},
            {"TrainDays": 14, "Season": "Winter", "Date": "2025-01-16", "Target": "foF2", "Model": "RandomForest", "R2": 0.1, "MAE": 9.9},
            {"TrainDays": 21, "Season": "Spring", "Date": "2025-02-15", "Target": "foF2", "Model": "RandomForest", "R2": 0.6, "MAE": 1.2},
        ]
    )
    phys_df = pd.DataFrame(
        [
            {"TrainDays": 21, "Season": "Winter", "Date": "2025-01-18", "Target": "foF2", "Model": "IRI", "R2": 0.5, "MAE": 1.5},
            {"TrainDays": 21, "Season": "Spring", "Date": "2025-02-18", "Target": "foF2", "Model": "IRI", "R2": 0.4, "MAE": 1.7},
        ]
    )
    config = pipeline.PipelineConfig(
        plot_train_days=21,
        ml_date_start="2025-01-01",
        ml_date_end="2025-01-31",
    )

    comparison_df = pipeline.build_metric_comparison_table(
        metrics_df,
        phys_df,
        config=config,
        range_kind="month",
    )

    assert set(comparison_df["RangeKind"]) == {"month"}
    assert set(comparison_df["RangeLabel"]) == {"2025-01"}
    assert set(comparison_df["Model"]) == {"RandomForest", "IRI"}
    rf_row = comparison_df[comparison_df["Model"] == "RandomForest"].iloc[0]
    assert np.isclose(rf_row["R2"], 0.8)
