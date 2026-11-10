"""Descriptive checks for the development data; no fitted model or backtest."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import unicodedata
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEVELOPMENT_START = pd.Timestamp("2015-01-01")
DEVELOPMENT_END = pd.Timestamp("2024-12-31")
ENTITY_PATTERNS = {
    "SBER": r"\bсбер(?:банк\w*)?\b|\bsberp?\b",
    "GAZP": r"\bгазпром\b|\bgazp\b",
    "LKOH": r"\bлукойл\w*\b|\blkoh\b",
    "GMKN": r"\bнорникел\w*\b|\bнорильск\w*\s+никел\w*\b|\bgmkn\b",
    "NVTK": r"\bноватэк\w*\b|\bnvtk\b",
    "ROSN": r"\bроснефт\w*\b|\brosn\b",
    "MGNT": r"\bмагнит\b|\bmgnt\b",
    "YNDX_YDEX": r"\bяндекс\w*\b|\byndx\b|\bydex\b",
    "TCSG_T": r"\bтинькофф\w*\b|\btcs[g]?\b|\bт[- ]банк\w*\b",
    "FIVE": r"\bx5\b|\bfive\b|\bпят[её]рочк\w*\b|\bперекр[её]сток\w*\b",
}
ENTITY_TICKERS = {
    "YNDX_YDEX": ["YNDX", "YDEX"],
    "TCSG_T": ["TCSG", "T"],
    "FIVE": ["FIVE"],
}


def parse_dates(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", format="mixed").dt.normalize()


def validate_development_dates(values: pd.Series, name: str) -> pd.Series:
    dates = parse_dates(values)
    if (dates > DEVELOPMENT_END).any():
        raise ValueError(f"{name} contains holdout dates after 2024-12-31")
    if (dates < DEVELOPMENT_START).any():
        raise ValueError(f"{name} contains dates before development start 2015-01-01")
    return dates


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def price_targets(
    open_prices: pd.DataFrame, calendar: pd.DatetimeIndex, horizon: int
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Open(t+1+h)/open(t+1), with an exact common-calendar alignment.

    Missing observations and calendar closures longer than seven days invalidate
    the diagnostic label. Large raw jumps remain visible and are flagged, since
    an unadjusted split cannot be distinguished from a loss by price alone.
    """
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if not calendar.is_unique or not calendar.is_monotonic_increasing:
        raise ValueError("calendar must be unique and increasing")
    if not open_prices.index.is_unique:
        raise ValueError("open price dates must be unique")
    prices = open_prices.reindex(calendar).astype(float)
    valid = np.isfinite(prices) & prices.gt(0)
    complete = valid.shift(-1, fill_value=False).copy()
    for offset in range(2, horizon + 2):
        complete &= valid.shift(-offset, fill_value=False)
    gaps = pd.Series(calendar, index=calendar).diff().dt.days.gt(7)
    gap_span = pd.Series(False, index=calendar)
    for offset in range(1, horizon + 2):
        gap_span |= gaps.shift(-offset, fill_value=False)
    gap_flags = pd.DataFrame(
        np.repeat(gap_span.to_numpy()[:, None], prices.shape[1], axis=1),
        index=calendar,
        columns=prices.columns,
    )
    jumps = prices.pct_change(fill_method=None).abs().gt(0.5)
    jump_span = jumps.copy() & False
    for offset in range(1, horizon + 2):
        jump_span |= jumps.shift(-offset, fill_value=False)
    targets = prices.shift(-(horizon + 1)).div(prices.shift(-1)).sub(1)
    targets = targets.where(complete & ~gap_flags)
    return targets, {
        "missing_price": ~complete,
        "calendar_gap": gap_flags,
        "large_raw_jump": jump_span,
    }


def profile_market(candles: pd.DataFrame, calendar: pd.DatetimeIndex) -> dict:
    frame = candles.copy()
    frame["date"] = validate_development_dates(frame["begin"], "candles")
    numeric = ["open", "close", "high", "low", "value", "volume"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    price = frame[["open", "close", "high", "low"]]
    complete = price.notna().all(axis=1)
    invalid_ohlc = complete & (
        frame["low"].gt(frame["high"])
        | frame["open"].lt(frame["low"])
        | frame["open"].gt(frame["high"])
        | frame["close"].lt(frame["low"])
        | frame["close"].gt(frame["high"])
    )
    duplicates = int(frame.duplicated(["secid", "date"]).sum())
    if duplicates:
        raise ValueError(f"candles contains {duplicates} duplicate security/date keys")
    if frame["date"].isna().any():
        raise ValueError("candles contains unparseable dates")
    outside_calendar = ~frame["date"].isin(calendar)
    coverage = []
    for ticker, group in frame.groupby("secid", sort=True):
        start, end = group["date"].min(), group["date"].max()
        observed = calendar[(calendar >= start) & (calendar <= end)]
        actual = pd.DatetimeIndex(group["date"])
        coverage.append(
            {
                "secid": ticker,
                "rows": len(group),
                "first_date": start.strftime("%Y-%m-%d"),
                "last_date": end.strftime("%Y-%m-%d"),
                "benchmark_dates_within_span": len(observed),
                "absent_on_benchmark_date": len(observed.difference(actual)),
                "zero_volume_rows": int(group["volume"].eq(0).sum()),
            }
        )
    quality = {
        "rows": len(frame),
        "securities": frame["secid"].nunique(),
        "date_min": frame["date"].min().strftime("%Y-%m-%d"),
        "date_max": frame["date"].max().strftime("%Y-%m-%d"),
        "duplicate_security_date_keys": duplicates,
        "ohlc_violations": int(invalid_ohlc.sum()),
        "nonpositive_prices": int(price.le(0).any(axis=1).sum()),
        "nonfinite_prices": int((~np.isfinite(price)).any(axis=1).sum()),
        "zero_volume_rows": int(frame["volume"].eq(0).sum()),
        "zero_value_rows": int(frame["value"].eq(0).sum()),
        "negative_value_rows": int(frame["value"].lt(0).sum()),
        "negative_volume_rows": int(frame["volume"].lt(0).sum()),
        "dates_outside_benchmark_calendar": int(outside_calendar.sum()),
        "nulls": {c: int(frame[c].isna().sum()) for c in numeric},
        "dtypes": {c: str(t) for c, t in frame.dtypes.items()},
    }
    return {"frame": frame, "quality": quality, "coverage": pd.DataFrame(coverage)}


def profile_news(news: pd.DataFrame) -> dict:
    frame = news.copy()
    frame["parsed_date"] = validate_development_dates(frame["date"], "news")
    for column in ["title", "body", "time", "tags", "source"]:
        frame[column] = frame[column].fillna("").astype(str)
    normalized = (
        frame["title"]
        .map(normalize_text)
        .str.cat(frame["body"].map(normalize_text), sep="\n")
    )
    frame["text_hash"] = normalized.map(
        lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest()
    )
    frame["text_length"] = frame["body"].str.len()
    frame["source"] = frame["source"].map(normalize_text)
    time_pattern = r"(?:[01]\d|2[0-3]):[0-5]\d(?::[0-5]\d)?"
    valid_times = frame["time"].str.fullmatch(time_pattern)
    frame["valid_time"] = valid_times
    blank = {
        column: int(frame[column].str.strip().eq("").sum())
        for column in ["title", "body", "time", "tags", "source"]
    }
    hash_split_counts = frame.groupby("text_hash")["dataset_split"].nunique()
    cross_split_hashes = set(hash_split_counts[hash_split_counts > 1].index)
    quality = {
        "rows": len(frame),
        "date_min": frame["parsed_date"].min().strftime("%Y-%m-%d"),
        "date_max": frame["parsed_date"].max().strftime("%Y-%m-%d"),
        "invalid_dates": int(frame["parsed_date"].isna().sum()),
        "missing_or_empty": blank,
        "invalid_or_empty_time": int((~valid_times).sum()),
        "placeholder_counts": {
            "time_not_stated": int(
                frame["time"].str.strip().str.casefold().eq("not stated").sum()
            ),
            "tags_empty_list": int(frame["tags"].str.strip().eq("[]").sum()),
        },
        "exact_duplicate_rows": int(
            news.duplicated(subset=[c for c in news if c != "dataset_split"]).sum()
        ),
        "normalized_text_duplicate_rows": int(frame["text_hash"].duplicated().sum()),
        "cross_split_duplicate_hashes": len(cross_split_hashes),
        "cross_split_duplicate_rows": int(
            frame["text_hash"].isin(cross_split_hashes).sum()
        ),
        "sources": {k: int(v) for k, v in frame["source"].value_counts().items()},
        "dataset_splits": {
            k: int(v) for k, v in frame["dataset_split"].value_counts().items()
        },
        "body_length_quantiles": {
            str(k): float(v)
            for k, v in frame["text_length"]
            .quantile([0, 0.25, 0.5, 0.75, 0.99, 1])
            .items()
        },
        "columns": list(news.columns),
        "dtypes": {c: str(t) for c, t in news.dtypes.items()},
        "sentiment_label_columns": [
            c for c in news.columns if re.search(r"sentiment|label|тональ", c, re.I)
        ],
    }
    frame = frame.sort_values(["parsed_date", "time"], kind="stable")
    distinct = frame.drop_duplicates("text_hash")
    searchable = (
        distinct["title"]
        .str.cat(distinct["body"], sep=" ")
        .str.cat(distinct["tags"], sep=" ")
        .map(normalize_text)
    )
    hits = pd.DataFrame(
        {
            name: searchable.str.contains(pattern, regex=True)
            for name, pattern in ENTITY_PATTERNS.items()
        },
        index=distinct.index,
    )
    return {"frame": frame, "distinct": distinct, "hits": hits, "quality": quality}


def _save_figure(fig: plt.Figure, output: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(output / "figures" / f"{name}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def _write_csv(frame: pd.DataFrame, output: Path, name: str) -> None:
    frame.to_csv(output / f"{name}.csv", index=False, float_format="%.8g")


def alias_overlap(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    values = ["open", "high", "low", "close", "value", "volume"]
    for first, second in [
        ("TCSG", "T"),
        ("YNDX", "YDEX"),
        ("RUALR", "RUAL"),
        ("EONR", "UPRO"),
        ("FIVE", "X5"),
    ]:
        a = frame.loc[frame["secid"].eq(first), ["date", *values]]
        b = frame.loc[frame["secid"].eq(second), ["date", *values]]
        common = a.merge(b, on="date", suffixes=("_a", "_b"))
        equal = np.ones(len(common), dtype=bool)
        for column in values:
            equal &= np.isclose(
                common[f"{column}_a"],
                common[f"{column}_b"],
                rtol=0,
                atol=1e-8,
                equal_nan=True,
            )
        ratios = (
            common["close_a"]
            .div(common["close_b"])
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        record = {
            "first_code": first,
            "second_code": second,
            "first_rows": len(a),
            "second_rows": len(b),
            "common_dates": len(common),
            "equal_ohlcv_rows": int(equal.sum()),
            "common_date_min": common["date"].min().strftime("%Y-%m-%d")
            if len(common)
            else None,
            "common_date_max": common["date"].max().strftime("%Y-%m-%d")
            if len(common)
            else None,
            "close_ratio_median": float(ratios.median()) if len(ratios) else None,
            "close_ratio_min": float(ratios.min()) if len(ratios) else None,
            "close_ratio_max": float(ratios.max()) if len(ratios) else None,
            "near_constant_close_ratio": bool(
                np.isclose(ratios, ratios.median(), rtol=1e-9, atol=1e-9).all()
            )
            if len(ratios)
            else False,
        }
        rows.append(record)
    return pd.DataFrame(rows)


def monthly_news_counts(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame.dropna(subset=["parsed_date"])
    monthly = (
        valid.assign(month=valid["parsed_date"].dt.to_period("M"))
        .groupby(["month", "source"])
        .size()
        .unstack(fill_value=0)
    )
    if not monthly.empty:
        monthly = monthly.reindex(
            pd.period_range(monthly.index.min(), monthly.index.max(), freq="M"),
            fill_value=0,
        )
    monthly.index.name = "month"
    return monthly


def _number_summary(values: pd.Series) -> dict:
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p01": float(values.quantile(0.01)),
        "median": float(values.median()),
        "p99": float(values.quantile(0.99)),
        "max": float(values.max()),
    }


def load_verified_tables(processed: Path, config_path: Path | str | None) -> dict:
    collection = json.loads((processed / "collection.json").read_text())
    sources = json.loads(
        (processed.parent.parent / "reports/eda/sources_manifest.json").read_text()
    )
    config_hash = hashlib.sha256(
        json.dumps(collection["config"], sort_keys=True).encode()
    ).hexdigest()
    if config_path is not None:
        requested = json.loads(Path(config_path).read_text())
        if collection["config"] != requested:
            raise ValueError(
                "Processed snapshot config differs from the requested config"
            )
    if any(
        record.get("config_sha256") != config_hash for record in [collection, sources]
    ):
        raise ValueError("Snapshot config checksum mismatch")
    source_records = sorted(
        [
            {
                "url": record["url"],
                "sha256": record["sha256"],
                "revision": record.get("revision"),
            }
            for record in sources["sources"]
        ],
        key=lambda record: record["url"],
    )
    fingerprint = hashlib.sha256(
        json.dumps(source_records, sort_keys=True).encode()
    ).hexdigest()
    if any(
        record.get("source_fingerprint") != fingerprint
        for record in [collection, sources]
    ):
        raise ValueError("Processed snapshot and source manifest fingerprints differ")
    tables = {}
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
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != collection["files"][name]["sha256"]:
            raise ValueError(
                f"Processed {name} checksum mismatch; recollect the snapshot"
            )
        tables[name] = pd.read_parquet(target)
    return tables


def execute_notebook(notebook, root: Path) -> None:
    """Execute notebook cells with real IPython display capture, without sockets."""
    import nbformat
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    from traitlets.config import Config

    shell = InteractiveShell.instance(
        config=Config({"HistoryManager": {"hist_file": ":memory:"}}), user_ns={}
    )
    original_directory = Path.cwd()
    count = 0
    try:
        os.chdir(root)
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            count += 1
            with capture_output(display=True) as captured:
                result = shell.run_cell(cell.source, store_history=True)
            error = result.error_before_exec or result.error_in_exec
            if error is not None:
                raise RuntimeError(f"Notebook cell {count} failed: {error}") from error
            outputs = []
            for name in ["stdout", "stderr"]:
                if text := getattr(captured, name):
                    outputs.append(
                        nbformat.v4.new_output("stream", name=name, text=text)
                    )
            for rich in captured.outputs:
                outputs.append(
                    nbformat.v4.new_output(
                        "display_data", data=rich.data, metadata=rich.metadata
                    )
                )
            cell.outputs = outputs
            cell.execution_count = count
        notebook.metadata["execution"] = {
            "executor": "ipython-inprocess",
            "python_version": platform.python_version(),
        }
    finally:
        os.chdir(original_directory)
        shell.reset(new_session=False)
        InteractiveShell.clear_instance()


def build_report(
    processed_dir: Path | str,
    output_dir: Path | str,
    *,
    config_path: Path | str | None = None,
) -> dict:
    processed, output = Path(processed_dir), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    tables = load_verified_tables(processed, config_path)
    benchmark = tables["benchmark"].copy()
    benchmark["date"] = validate_development_dates(benchmark["tradedate"], "benchmark")
    if benchmark["date"].isna().any() or benchmark["date"].duplicated().any():
        raise ValueError("benchmark calendar has invalid or duplicate dates")
    calendar = pd.DatetimeIndex(benchmark["date"].sort_values().unique())
    market = profile_market(tables["candles"], calendar)
    frame = market["frame"]
    aliases = alias_overlap(frame)
    _write_csv(aliases, output, "alias_overlap")
    snapshots = tables["index_snapshots"]
    all_tickers = sorted(set(snapshots["ticker"]) | set(frame["secid"]))
    empty_tickers = sorted(set(snapshots["ticker"]) - set(frame["secid"]))
    coverage = market["coverage"].set_index("secid").reindex(all_tickers)
    coverage["rows"] = coverage["rows"].fillna(0).astype(int)
    _write_csv(coverage.reset_index(), output, "security_coverage")
    market["quality"]["empty_candle_tickers"] = empty_tickers
    for column in ["requested_date", "tradedate"]:
        validate_development_dates(snapshots[column], f"index.{column}")
    snapshot_summary = (
        snapshots.groupby(["requested_date", "tradedate"])
        .agg(securities=("ticker", "nunique"))
        .reset_index()
    )
    _write_csv(snapshot_summary, output, "index_snapshot_coverage")

    annual = (
        frame.assign(year=frame["date"].dt.year)
        .groupby(["secid", "year"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=all_tickers, fill_value=0)
        .reindex(columns=range(2015, 2025), fill_value=0)
    )
    _write_csv(annual.reset_index(), output, "annual_observations")
    fig, ax = plt.subplots(figsize=(10, max(8, len(annual) * 0.16)))
    im = ax.imshow(annual.to_numpy(), aspect="auto", cmap="YlGnBu", vmin=0)
    ax.set(
        xticks=range(10),
        xticklabels=annual.columns,
        yticks=range(len(annual)),
        yticklabels=annual.index,
        title="Daily candle observations by security and year",
    )
    ax.tick_params(axis="y", labelsize=7)
    fig.colorbar(im, ax=ax, label="Observed days")
    _save_figure(fig, output, "coverage")

    daily = (
        frame.groupby("date")
        .agg(securities=("secid", "nunique"), total_value=("value", "sum"))
        .reindex(pd.date_range("2022-01-01", "2022-05-31"))
    )
    daily.index.name = "date"
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].bar(daily.index, daily["securities"].fillna(0), width=1)
    axes[0].set(
        ylabel="Securities", title="Observed candle activity around the 2022 closure"
    )
    axes[1].bar(daily.index, daily["total_value"].fillna(0) / 1e9, width=1)
    axes[1].set(ylabel="Turnover, RUB bn", xlabel="Calendar date")
    _save_figure(fig, output, "activity_2022")
    _write_csv(daily.reset_index(), output, "activity_2022")

    opens = frame.pivot(index="date", columns="secid", values="open").reindex(calendar)
    closes = frame.pivot(index="date", columns="secid", values="close").reindex(
        calendar
    )
    daily_returns = closes.pct_change(fill_method=None)
    daily_returns = daily_returns.where(closes.gt(0) & closes.shift(1).gt(0))
    long_gaps = pd.Series(calendar, index=calendar).diff().dt.days.gt(7)
    daily_returns.loc[long_gaps] = np.nan
    target_summary = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, horizon in zip(axes, [5, 20], strict=True):
        targets, flags = price_targets(opens, calendar, horizon)
        clean = targets.mask(flags["large_raw_jump"])
        raw_values = targets.stack()
        clean_values = clean.stack()
        quality = _number_summary(raw_values)
        quality.update(
            horizon=horizon,
            grid_rows=int(targets.size),
            available_share=float(targets.notna().to_numpy().mean()),
            missing_price_grid_rows=int(flags["missing_price"].to_numpy().sum()),
            calendar_gap_grid_rows=int(flags["calendar_gap"].to_numpy().sum()),
            large_jump_grid_rows=int(flags["large_raw_jump"].to_numpy().sum()),
            clean_count=len(clean_values),
            clean_values_clipped_for_display=int(clean_values.abs().gt(0.5).sum()),
        )
        target_summary.append(quality)
        ax.hist(clean_values.clip(-0.5, 0.5), bins=80, alpha=0.8)
        ax.set(
            title=f"h={horizon}: diagnostic price return",
            xlabel="Return (clipped for display)",
            ylabel="Security/date labels",
        )
    _save_figure(fig, output, "price_targets")
    _write_csv(pd.DataFrame(target_summary), output, "target_summary")

    regime_rows = []
    regimes = {
        "2015-2021": daily_returns.index < "2022-01-01",
        "2022-2024": daily_returns.index >= "2022-01-01",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, mask in regimes.items():
        values = daily_returns.loc[mask].stack()
        summary = _number_summary(values)
        summary["regime"] = name
        summary["abs_return_gt_50pct"] = int(values.abs().gt(0.5).sum())
        summary["values_clipped_for_display"] = int(values.abs().gt(0.15).sum())
        summary["median_20d_volatility"] = float(
            daily_returns.loc[mask].rolling(20, min_periods=15).std().stack().median()
        )
        dates = calendar[mask]
        summary["median_daily_turnover_rub"] = float(
            frame.loc[frame["date"].isin(dates), "value"].median()
        )
        regime_rows.append(summary)
        axes[0].hist(
            values.clip(-0.15, 0.15), bins=80, density=True, alpha=0.5, label=name
        )
    axes[0].set(
        title="As-served close-to-close price returns",
        xlabel="Return (display clipped)",
        ylabel="Density",
    )
    axes[0].legend()
    median_turnover = (
        frame.assign(year=frame["date"].dt.year).groupby("year")["value"].median()
    )
    axes[1].plot(median_turnover.index, median_turnover / 1e6, marker="o")
    axes[1].set(
        title="Median daily turnover per observed security",
        ylabel="RUB mn",
        xlabel="Year",
    )
    _save_figure(fig, output, "regime_returns")
    _write_csv(pd.DataFrame(regime_rows), output, "regime_summary")

    corr_names = [
        c
        for c in [
            "SBER",
            "GAZP",
            "LKOH",
            "GMKN",
            "NVTK",
            "ROSN",
            "MGNT",
            "MOEX",
            "CHMF",
            "MTSS",
        ]
        if c in closes
    ]
    corr_returns = daily_returns[corr_names].mask(daily_returns[corr_names].abs() > 0.5)
    corr = corr_returns.corr(min_periods=252)
    overlap = corr_returns.notna().astype(int).T @ corr_returns.notna().astype(int)
    _write_csv(corr.rename_axis("secid").reset_index(), output, "correlation")
    _write_csv(
        overlap.rename_axis("secid").reset_index(), output, "correlation_overlap"
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set(
        xticks=range(len(corr)),
        xticklabels=corr_names,
        yticks=range(len(corr)),
        yticklabels=corr_names,
        title="Price-return correlation; at least 252 paired days",
    )
    ax.tick_params(axis="x", rotation=45)
    fig.colorbar(im, ax=ax)
    _save_figure(fig, output, "correlation")

    history = tables["history"].copy()
    history_summary = {}
    if not history.empty:
        history["date"] = validate_development_dates(history["tradedate"], "history")
        key = ["secid", "date"] + (
            ["tradingsession"] if "tradingsession" in history else []
        )
        history_summary["duplicate_natural_keys"] = int(history.duplicated(key).sum())
        history_summary["rows"] = len(history)
        history_summary["missing_legalcloseprice"] = int(
            history["legalcloseprice"].isna().sum()
        )
        history_summary["missing_open"] = int(history["open"].isna().sum())
        matched = history.merge(
            frame[["secid", "date", "close"]],
            on=["secid", "date"],
            how="inner",
            suffixes=("_history", "_candle"),
        )
        comparable = matched[["legalcloseprice", "close_candle"]].notna().all(axis=1)
        different = comparable & ~np.isclose(
            matched["legalcloseprice"], matched["close_candle"], equal_nan=True
        )
        history_summary["matched_candle_rows"] = len(matched)
        history_summary["comparable_legalclose_candle_rows"] = int(comparable.sum())
        history_summary["different_legalclose_candle_rows"] = int(different.sum())
        history_ranges = (
            history.groupby("secid")
            .agg(
                rows=("date", "size"),
                date_min=("date", "min"),
                date_max=("date", "max"),
            )
            .reset_index()
        )
        _write_csv(history_ranges, output, "history_security_summary")

    news = profile_news(tables["news"])
    source_summary = (
        news["frame"]
        .groupby("source")
        .agg(
            rows=("text_hash", "size"),
            date_min=("parsed_date", "min"),
            date_max=("parsed_date", "max"),
            parseable_times=("valid_time", "sum"),
        )
        .reset_index()
    )
    _write_csv(source_summary, output, "news_source_summary")
    daily_peaks = (
        news["frame"]
        .groupby(["parsed_date", "source"])
        .size()
        .nlargest(10)
        .rename("rows")
        .reset_index()
    )
    daily_peaks["parsed_date"] = daily_peaks["parsed_date"].dt.strftime("%Y-%m-%d")
    _write_csv(daily_peaks, output, "news_source_date_peaks")
    news["quality"]["largest_source_day_groups"] = daily_peaks.to_dict("records")
    split_summary = (
        news["frame"]
        .groupby("dataset_split")
        .agg(
            rows=("text_hash", "size"),
            date_min=("parsed_date", "min"),
            date_max=("parsed_date", "max"),
        )
        .reset_index()
    )
    _write_csv(split_summary, output, "news_split_summary")
    collection = json.loads((processed / "collection.json").read_text())
    news["quality"]["original_corpus"] = {
        key: collection.get(key)
        for key in [
            "news_original_rows",
            "news_rows_outside_development",
            "news_original_date_min",
            "news_original_date_max",
            "original_split_summary",
        ]
    }
    news["quality"]["dataset_card_claims"] = {
        "prose_article_count": 92377,
        "embedded_split_rows": 73564 + 18391,
        "source": "https://huggingface.co/datasets/Kasymkhan/RussianFinancialNews/blob/f83da5c05c82c9086806cf90ded4155052f9605c/README.md",
    }
    monthly = monthly_news_counts(news["frame"])
    news["quality"]["calendar_months_in_span"] = len(monthly)
    news["quality"]["months_without_articles"] = int(monthly.sum(axis=1).eq(0).sum())
    yearly = (
        news["frame"]
        .assign(year=news["frame"]["parsed_date"].dt.year)
        .groupby("year")
        .size()
        .rename("rows")
        .reset_index()
    )
    _write_csv(yearly, output, "news_yearly")
    news["quality"]["yearly_rows"] = yearly.to_dict("records")
    _write_csv(monthly.rename_axis("month").reset_index(), output, "news_monthly")
    fig, ax = plt.subplots(figsize=(11, 5))
    monthly.index = monthly.index.to_timestamp()
    monthly.plot.area(ax=ax, linewidth=0)
    ax.set(
        title="RussianFinancialNews: article count by month and source",
        ylabel="Articles",
        xlabel="Publication month",
    )
    _save_figure(fig, output, "news_monthly")
    news_coverage = []
    unique_news = news["distinct"]
    for entity in ENTITY_PATTERNS:
        tickers = ENTITY_TICKERS.get(entity, [entity])
        entity_candles = frame[frame["secid"].isin(tickers)]
        hit_news = unique_news.loc[news["hits"][entity]]
        if entity_candles.empty:
            continue
        start = max(entity_candles["date"].min(), unique_news["parsed_date"].min())
        end = min(entity_candles["date"].max(), unique_news["parsed_date"].max())
        overlap_days = pd.date_range(start, end)
        trading_days = calendar[(calendar >= start) & (calendar <= end)]
        dates_with_news = pd.DatetimeIndex(hit_news["parsed_date"].dropna().unique())
        news_coverage.append(
            {
                "entity": entity,
                "matched_distinct_articles": len(hit_news),
                "overlap_start": start.strftime("%Y-%m-%d"),
                "overlap_end": end.strftime("%Y-%m-%d"),
                "overlap_calendar_days": len(overlap_days),
                "overlap_benchmark_days": len(trading_days),
                "calendar_days_with_news": len(
                    overlap_days.intersection(dates_with_news)
                ),
                "benchmark_days_with_news": len(
                    trading_days.intersection(dates_with_news)
                ),
                "benchmark_day_coverage": len(
                    trading_days.intersection(dates_with_news)
                )
                / len(trading_days)
                if len(trading_days)
                else None,
            }
        )
    news_coverage_frame = pd.DataFrame(news_coverage)
    _write_csv(news_coverage_frame, output, "news_entity_coverage")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(
        news_coverage_frame["entity"],
        news_coverage_frame["benchmark_day_coverage"] * 100,
    )
    ax.set(
        title="Weak company matching: news on observed benchmark days",
        ylabel="Days with at least one article, %",
    )
    ax.tick_params(axis="x", rotation=35)
    _save_figure(fig, output, "news_coverage")

    macro_summary = {}
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, table_name, value_column, label in zip(
        axes,
        ["fx", "key_rate"],
        ["usd_rub", "rate"],
        ["USD/RUB (official)", "CBR key rate, %"],
        strict=True,
    ):
        table = tables[table_name].copy()
        table["date"] = validate_development_dates(table["date"], table_name)
        table = table.sort_values("date")
        numeric = pd.to_numeric(table[value_column], errors="coerce")
        macro_summary[table_name] = {
            "rows": len(table),
            "duplicate_dates": int(table["date"].duplicated().sum()),
            "invalid_dates": int(table["date"].isna().sum()),
            "missing_values": int(numeric.isna().sum()),
            "nonpositive_values": int(numeric.le(0).sum()),
            "date_min": table["date"].min().strftime("%Y-%m-%d"),
            "date_max": table["date"].max().strftime("%Y-%m-%d"),
            "benchmark_days_before_first_observation": int(
                (calendar < table["date"].min()).sum()
            ),
        }
        ax.plot(table["date"], numeric)
        ax.set(ylabel=label)
    axes[0].set_title(
        "Macro source coverage; effective dates are not publication timestamps"
    )
    _save_figure(fig, output, "macro")

    manifest = {
        "development_window": ["2015-01-01", "2024-12-31"],
        "config": json.loads(Path(config_path).read_text()) if config_path else None,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["pandas", "numpy", "matplotlib", "pyarrow"]
        },
        "tables": {
            name: {
                "rows": len(table),
                "columns": len(table.columns),
                "processed_bytes": (processed / f"{name}.parquet").stat().st_size,
            }
            for name, table in tables.items()
        },
        "market": market["quality"],
        "alias_overlap": aliases.astype(object)
        .where(aliases.notna(), None)
        .to_dict("records"),
        "index_snapshots": {
            "requested_snapshots": len(snapshot_summary),
            "unique_tickers": int(snapshots["ticker"].nunique()),
            "rows": len(snapshots),
        },
        "benchmark": {
            "observed_dates": len(calendar),
            "date_min": calendar.min().strftime("%Y-%m-%d"),
            "date_max": calendar.max().strftime("%Y-%m-%d"),
        },
        "targets": target_summary,
        "regimes": regime_rows,
        "history": history_summary,
        "news": news["quality"],
        "macro": macro_summary,
        "limitations": [
            "Quarterly snapshots define a diagnostic panel, not a daily point-in-time investable universe.",
            "Daily candles may cover several sessions; they do not establish the primary-session signal cutoff.",
            "Security codes are not independent economic instruments; overlapping aliases and retrospectively rescaled candle histories require identity resolution.",
            "Targets use prices as served retrospectively by ISS; corporate-action adjustments have not been independently verified and total return is not established.",
            "Observed MCFTR dates are a proxy calendar, not an independently validated exchange calendar.",
            "Historical available_at and news timezone have not been established by this retrospective download.",
            "Company regex matches are weak labels; duplicate detection does not measure near-duplicate syndication.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    findings = f"""# Результаты вычислений EDA

Период разработки: 2015–2024. Holdout 2025–2026 не исследовался.

- Дневные свечи: {market["quality"]["rows"]:,} строк, {market["quality"]["securities"]} тикеров; нарушения OHLC: {market["quality"]["ohlc_violations"]}.
- Срезы IMOEX: {len(snapshot_summary)} квартальных дат, {snapshots["ticker"].nunique()} уникальных тикеров. Полная ежедневная история состава индекса не восстановлена.
- Новости: {news["quality"]["rows"]:,} строк, {news["quality"]["date_min"]} — {news["quality"]["date_max"]}; повторных нормализованных текстов: {news["quality"]["normalized_text_duplicate_rows"]:,}.
- Временная метка новости не доказывает момент её доступности; часовой пояс и точность entity linking требуют отдельной проверки.
- Ретроспективные цены могут менять масштаб истории. Доходности здесь диагностические; дивиденды, сплиты и коэффициенты обмена независимо не проверены.

Графики и сводные таблицы воспроизводятся `python scripts/run_eda.py --offline`. Числа в `summary.json` относятся к текущему зафиксированному кэшу, а не к гарантированно полному рынку.
"""
    (output / "findings.md").write_text(findings)
    return manifest
