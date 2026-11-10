import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from moex_portfolio.eda import (
    alias_overlap,
    execute_notebook,
    load_verified_tables,
    monthly_news_counts,
    normalize_text,
    price_targets,
    profile_market,
    profile_news,
    validate_development_dates,
)


def test_target_uses_next_open_and_exact_horizon():
    calendar = pd.bdate_range("2024-01-01", periods=9)
    prices = pd.DataFrame({"A": np.arange(10, 19)}, index=calendar)
    labels, flags = price_targets(prices, calendar, 5)
    assert labels.iloc[0, 0] == pytest.approx(16 / 11 - 1)
    assert labels.iloc[-6:, 0].isna().all()
    assert not flags["calendar_gap"].any().any()


def test_missing_date_is_not_skipped_or_filled():
    calendar = pd.bdate_range("2024-01-01", periods=9)
    prices = pd.DataFrame({"A": np.arange(10, 19)}, index=calendar).drop(calendar[3])
    labels, flags = price_targets(prices, calendar, 5)
    assert np.isnan(labels.iloc[0, 0])
    assert flags["missing_price"].iloc[0, 0]


def test_long_closure_invalidates_label_even_with_positive_prices():
    calendar = pd.DatetimeIndex(
        ["2022-02-24", "2022-02-25", "2022-03-24", "2022-03-25", "2022-03-28"]
    )
    prices = pd.DataFrame({"A": [100, 80, 60, 65, 70]}, index=calendar)
    labels, flags = price_targets(prices, calendar, 1)
    assert np.isnan(labels.iloc[0, 0])
    assert flags["calendar_gap"].iloc[0, 0]
    assert np.isnan(labels.iloc[1, 0])
    assert labels.iloc[2, 0] == pytest.approx(70 / 65 - 1)


def test_raw_corporate_jump_is_flagged_not_silently_adjusted():
    calendar = pd.bdate_range("2024-04-01", periods=4)
    prices = pd.DataFrame({"A": [15000, 15000, 150, 160]}, index=calendar)
    labels, flags = price_targets(prices, calendar, 1)
    assert labels.iloc[0, 0] == pytest.approx(-0.99)
    assert flags["large_raw_jump"].iloc[0, 0]


def test_nonpositive_price_and_duplicate_calendar_fail_safely():
    calendar = pd.bdate_range("2024-01-01", periods=4)
    prices = pd.DataFrame({"A": [10, 0, 20, 30]}, index=calendar)
    labels, _ = price_targets(prices, calendar, 1)
    assert np.isnan(labels.iloc[0, 0])
    with pytest.raises(ValueError, match="unique"):
        price_targets(prices, calendar.append(calendar[-1:]), 1)


def test_holdout_is_rejected_before_analysis():
    with pytest.raises(ValueError, match="holdout"):
        validate_development_dates(pd.Series(["2024-12-31", "2025-01-01"]), "test")
    with pytest.raises(ValueError, match="before development"):
        validate_development_dates(pd.Series(["2014-12-31"]), "test")


def test_news_normalization_detects_cross_split_duplicates():
    news = pd.DataFrame(
        {
            "title": ["  Сбер вырос", "сбер ВЫРОС ", "Другое"],
            "body": ["Текст\nновости", "Текст новости", ""],
            "date": ["2024-01-01", "2024-01-02", "bad"],
            "time": ["12:00", "25:99", ""],
            "tags": ["SBER", "", ""],
            "source": ["RBC", "rbc", "rbc"],
            "dataset_split": ["train", "test", "train"],
        }
    )
    result = profile_news(news)
    assert result["quality"]["normalized_text_duplicate_rows"] == 1
    assert result["quality"]["cross_split_duplicate_hashes"] == 1
    assert result["quality"]["invalid_or_empty_time"] == 2
    assert result["quality"]["invalid_dates"] == 1
    assert len(result["distinct"]) == 2
    assert result["hits"]["SBER"].sum() == 1
    assert normalize_text("Ａ  B\n C") == "a b c"


def test_market_flags_ohlc_and_does_not_hide_duplicate_keys():
    calendar = pd.bdate_range("2024-01-01", periods=2)
    frame = pd.DataFrame(
        {
            "secid": ["A", "A"],
            "begin": calendar,
            "open": [10, 20],
            "close": [11, 22],
            "low": [9, 19],
            "high": [12, 21],
            "value": [100, 0],
            "volume": [10, 0],
        }
    )
    result = profile_market(frame, calendar)
    assert result["quality"]["ohlc_violations"] == 1
    assert result["quality"]["zero_volume_rows"] == 1
    with pytest.raises(ValueError, match="duplicate"):
        profile_market(pd.concat([frame, frame.iloc[:1]]), calendar)


def make_processed_snapshot(tmp_path):
    processed = tmp_path / "data/processed"
    processed.mkdir(parents=True)
    reports = tmp_path / "reports/eda"
    reports.mkdir(parents=True)
    config = {"start": "2015-01-01", "end": "2024-12-31"}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    metadata = {}
    for name in [
        "candles",
        "index_snapshots",
        "history",
        "benchmark",
        "fx",
        "key_rate",
        "news",
    ]:
        target = processed / f"{name}.parquet"
        pd.DataFrame({"value": [1]}).to_parquet(target)
        metadata[name] = {"sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    config_sha = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    fingerprint = hashlib.sha256(json.dumps([], sort_keys=True).encode()).hexdigest()
    collection = {
        "config": config,
        "config_sha256": config_sha,
        "source_fingerprint": fingerprint,
        "files": metadata,
    }
    (processed / "collection.json").write_text(json.dumps(collection))
    (reports / "sources_manifest.json").write_text(
        json.dumps(
            {
                "config_sha256": config_sha,
                "source_fingerprint": fingerprint,
                "sources": [],
            }
        )
    )
    return processed, config_path


def test_processed_checksum_rejects_tampering(tmp_path):
    processed, config = make_processed_snapshot(tmp_path)
    assert len(load_verified_tables(processed, config)) == 7
    (processed / "news.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="news checksum mismatch"):
        load_verified_tables(processed, config)


def test_snapshot_configuration_and_source_identity_are_verified(tmp_path):
    processed, config = make_processed_snapshot(tmp_path)
    config.write_text(json.dumps({"start": "2020-01-01"}))
    with pytest.raises(ValueError, match="config differs"):
        load_verified_tables(processed, config)
    manifest = processed.parent.parent / "reports/eda/sources_manifest.json"
    sources = json.loads(manifest.read_text())
    sources["sources"] = [
        {"url": "https://example.org/data", "sha256": "changed", "revision": None}
    ]
    manifest.write_text(json.dumps(sources))
    with pytest.raises(ValueError, match="fingerprints differ"):
        load_verified_tables(processed, None)


def test_notebook_executes_cells_and_captures_real_outputs(tmp_path):
    import nbformat

    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell("value = 7"),
            nbformat.v4.new_code_cell(
                "import base64\nimport pandas as pd\n"
                "from IPython.display import Image, display\n"
                "display(pd.DataFrame({'computed': [value + 1]}))\n"
                "display(Image(data=base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a5XsAAAAASUVORK5CYII=')))\n"
                "print(value * 3)"
            ),
        ]
    )
    execute_notebook(notebook, tmp_path)
    assert [cell.execution_count for cell in notebook.cells] == [1, 2]
    assert notebook.metadata.execution.executor == "ipython-inprocess"
    output = notebook.cells[1].outputs
    assert any(item.output_type == "stream" and item.text == "21\n" for item in output)
    assert any(
        "text/html" in item.get("data", {})
        and "computed" in item.data["text/html"]
        and ">8<" in item.data["text/html"]
        for item in output
    )
    assert any("image/png" in item.get("data", {}) for item in output)


def test_notebook_execution_fails_on_cell_error(tmp_path):
    import nbformat

    notebook = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell("raise ValueError('actual failure')")]
    )
    with pytest.raises(RuntimeError, match="Notebook cell 1 failed: actual failure"):
        execute_notebook(notebook, tmp_path)


def test_alias_overlap_finds_rescaled_history_without_merging_it():
    days = pd.bdate_range("2024-01-01", periods=2)
    rows = []
    for code, scale in [("TCSG", 10), ("T", 1)]:
        for day in days:
            rows.append(
                {
                    "secid": code,
                    "date": day,
                    "open": 100 * scale,
                    "close": 101 * scale,
                    "low": 99 * scale,
                    "high": 102 * scale,
                    "value": 1000,
                    "volume": 10,
                }
            )
    result = alias_overlap(pd.DataFrame(rows)).iloc[0]
    assert result.common_dates == 2
    assert result.equal_ohlcv_rows == 0
    assert result.close_ratio_median == 10
    assert result.near_constant_close_ratio


def test_news_monthly_grid_keeps_zero_coverage_months():
    frame = pd.DataFrame(
        {
            "parsed_date": pd.to_datetime(["2024-01-01", "2024-03-01"]),
            "source": ["rbc", "rbc"],
        }
    )
    monthly = monthly_news_counts(frame)
    assert list(monthly.index.astype(str)) == ["2024-01", "2024-02", "2024-03"]
    assert monthly["rbc"].tolist() == [1, 0, 1]
