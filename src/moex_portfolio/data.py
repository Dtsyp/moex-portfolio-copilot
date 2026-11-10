"""Download a fixed development slice and retain verifiable source responses."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from io import BytesIO, StringIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

ISS = "https://iss.moex.com/iss"
DEVELOPMENT_END = date(2024, 12, 31)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


class CachedHTTP:
    """Reuse immutable URL responses; a broken cache is an error, never a refresh."""

    def __init__(self, root: Path, *, offline: bool = False, attempts: int = 4):
        if attempts < 1:
            raise ValueError("attempts must be positive")
        self.root = root
        self.offline = offline
        self.attempts = attempts
        self.records: dict[str, dict] = {}
        self.lock = threading.Lock()
        root.mkdir(parents=True, exist_ok=True)

    def get(self, url: str, source: str, revision: str | None = None) -> bytes:
        key = hashlib.sha256(url.encode()).hexdigest()
        target = self.root / f"{key}.bin"
        metadata = self.root / f"{key}.json"
        if target.exists() or metadata.exists():
            if not (target.exists() and metadata.exists()):
                raise ValueError(f"Incomplete cache entry: {key}")
            record = json.loads(metadata.read_text())
            payload = target.read_bytes()
            if (
                record["url"] != url
                or record["sha256"] != hashlib.sha256(payload).hexdigest()
                or record["bytes"] != len(payload)
                or record["revision"] != revision
            ):
                raise ValueError(f"Cache integrity check failed: {key}")
        else:
            if self.offline:
                raise FileNotFoundError(f"Offline cache miss: {url}")
            for attempt in range(self.attempts):
                try:
                    request = Request(
                        url, headers={"User-Agent": "moex-portfolio-research/0.1"}
                    )
                    with urlopen(request, timeout=60) as response:
                        payload = response.read()
                    break
                except HTTPError as error:
                    if error.code not in {429, 500, 502, 503, 504}:
                        raise
                    if attempt + 1 == self.attempts:
                        raise
                    retry_after = error.headers.get("Retry-After", "")
                    delay = float(retry_after) if retry_after.isdigit() else 2**attempt
                    time.sleep(min(delay, 30))
                except (URLError, TimeoutError, ConnectionError):
                    if attempt + 1 == self.attempts:
                        raise
                    time.sleep(2**attempt)
            record = {
                "source": source,
                "url": url,
                "path": str(target.relative_to(self.root.parent.parent)),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
                "fetched_at": datetime.now(UTC).isoformat(),
                "revision": revision,
            }
            temporary = target.with_suffix(".tmp")
            temporary.write_bytes(payload)
            temporary.replace(target)
            write_json(metadata, record)
        with self.lock:
            self.records[url] = record
        return payload


def iss_table(payload: dict, name: str) -> pd.DataFrame:
    block = payload[name]
    return pd.DataFrame(block["data"], columns=block["columns"]).rename(
        columns=str.lower
    )


def iss_pages(client: CachedHTTP, path: str, table: str, params: dict) -> pd.DataFrame:
    """Follow ISS cursor totals; candle endpoints end only on an empty page."""
    frames = []
    start = 0
    seen = set()
    previous_end = None
    expected_total = None
    while True:
        query = {**params, "iss.meta": "off", "start": start}
        query["iss.only"] = f"{table},{table}.cursor"
        url = f"{ISS}/{path}.json?{urlencode(query)}"
        payload = json.loads(client.get(url, "MOEX ISS"))
        frame = iss_table(payload, table)
        cursor_name = f"{table}.cursor"
        if cursor_name in payload:
            cursor = iss_table(payload, cursor_name).iloc[0]
            total = int(cursor["total"])
            if expected_total is not None and total != expected_total:
                raise ValueError(f"Changing ISS total while paging {path}")
            expected_total = total
            if int(cursor["index"]) != start:
                raise ValueError(f"Unexpected ISS cursor offset: {path}")
        if frame.empty:
            if expected_total is not None and start < expected_total:
                raise ValueError(f"Truncated ISS response: {path}")
            break
        signature = hashlib.sha256(
            json.dumps(payload[table]["data"]).encode()
        ).hexdigest()
        if signature in seen:
            raise ValueError(f"Repeated ISS page: {path}")
        seen.add(signature)
        if table == "candles":
            if (
                frame["begin"].duplicated().any()
                or not frame["begin"].is_monotonic_increasing
            ):
                raise ValueError(f"Unsorted candle response: {path}")
            first, last = frame["begin"].iloc[0], frame["begin"].iloc[-1]
            if previous_end is not None and first <= previous_end:
                raise ValueError(f"Non-progressing candle response: {path}")
            previous_end = last
        frames.append(frame)
        start += len(frame)
        if expected_total is not None:
            if start > expected_total:
                raise ValueError(f"ISS rows exceed cursor total: {path}")
            if start == expected_total:
                break
        if start > 100_000:
            raise ValueError(f"Unbounded ISS pagination: {path}")
    if not frames:
        return frame
    return pd.concat(frames, ignore_index=True)


def validate_config(config: dict) -> None:
    start, end = date.fromisoformat(config["start"]), date.fromisoformat(config["end"])
    holdout = date.fromisoformat(config["holdout_start"])
    history_start = date.fromisoformat(config["history_start"])
    if not (start <= history_start <= end <= DEVELOPMENT_END and end < holdout):
        raise ValueError("EDA dates must stay inside development history, before 2025")
    if not re.fullmatch(r"[0-9a-f]{40}", config["news_revision"]):
        raise ValueError("The news dataset must use an immutable commit revision")
    if not 1 <= config["workers"] <= 8:
        raise ValueError("workers must be between 1 and 8")


def index_snapshots(client: CachedHTTP, config: dict) -> pd.DataFrame:
    dates = pd.date_range(config["start"], config["end"], freq="QE")

    def snapshot(requested: pd.Timestamp) -> pd.DataFrame:
        actual = requested.date()
        for _ in range(10):
            frame = iss_pages(
                client,
                "statistics/engines/stock/markets/index/analytics/IMOEX",
                "analytics",
                {"date": actual.isoformat()},
            )
            if not frame.empty:
                returned = pd.to_datetime(frame["tradedate"])
                if (
                    returned.nunique() != 1
                    or not requested.date() - timedelta(days=9)
                    <= returned.iloc[0].date()
                    <= actual
                ):
                    raise ValueError("IMOEX snapshot returned an unexpected date")
                frame["requested_date"] = requested.date().isoformat()
                print(f"IMOEX {requested.date()}: {len(frame)} securities", flush=True)
                return frame
            actual -= timedelta(days=1)
        else:
            raise ValueError(f"No IMOEX snapshot near {requested.date()}")

    if dates.empty:
        raise ValueError("The configured period contains no quarterly snapshots")
    with ThreadPoolExecutor(config["workers"]) as pool:
        snapshots = list(pool.map(snapshot, dates))
    return pd.concat(snapshots, ignore_index=True)


def parse_fx(payload: bytes) -> pd.DataFrame:
    rows = []
    for record in ET.fromstring(payload).findall("Record"):
        nominal = int(record.findtext("Nominal"))
        value = float(record.findtext("Value").replace(",", "."))
        if nominal <= 0 or value <= 0:
            raise ValueError("Invalid CBR rate or nominal")
        rows.append(
            {
                "date": datetime.strptime(record.attrib["Date"], "%d.%m.%Y")
                .date()
                .isoformat(),
                "usd_rub": value / nominal,
                "nominal": nominal,
            }
        )
    if not rows:
        raise ValueError("No FX rates in CBR response")
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def parse_key_rate(payload: bytes) -> pd.DataFrame:
    tables = pd.read_html(
        StringIO(payload.decode("utf-8")), decimal=",", thousands=None
    )
    for table in tables:
        if len(table.columns) == 2 and "Дата" in str(table.columns[0]):
            result = pd.DataFrame(
                {
                    "date": pd.to_datetime(
                        table.iloc[:, 0], format="%d.%m.%Y"
                    ).dt.strftime("%Y-%m-%d"),
                    "rate": pd.to_numeric(table.iloc[:, 1]),
                }
            )
            if result.empty or result["rate"].isna().any():
                raise ValueError("Invalid CBR key-rate table")
            return result.sort_values("date").reset_index(drop=True)
    raise ValueError("CBR key-rate table was not found")


def check_dates(frame: pd.DataFrame, column: str, start: str, end: str) -> None:
    parsed = pd.to_datetime(frame[column], errors="raise")
    if not parsed.between(
        pd.Timestamp(start),
        pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1),
    ).all():
        raise ValueError(f"Source dates exceed requested development range: {column}")


def news_development_slice(
    news: pd.DataFrame, start: str, end: str
) -> tuple[pd.DataFrame, dict]:
    dates = pd.to_datetime(news["date"], format="mixed", errors="coerce").dt.normalize()
    within = dates.between(start, end)
    metadata = {
        "news_original_rows": len(news),
        "news_original_date_min": dates.min().isoformat(),
        "news_original_date_max": dates.max().isoformat(),
        "news_invalid_dates": int(dates.isna().sum()),
        "news_rows_before_development": int((dates < start).sum()),
        "news_rows_after_development": int((dates > end).sum()),
        "news_rows_outside_development": int((dates.notna() & ~within).sum()),
    }
    return news.loc[dates.isna() | within].reset_index(drop=True), metadata


def collect(root: Path, config: dict, *, offline: bool = False) -> dict:
    validate_config(config)
    client = CachedHTTP(root / "data/raw", offline=offline)
    processed = root / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    snapshots = index_snapshots(client, config)
    tickers = sorted(snapshots["ticker"].dropna().unique())
    params = {"from": config["start"], "till": config["end"]}

    def candles(ticker: str) -> pd.DataFrame:
        frame = iss_pages(
            client,
            f"engines/stock/markets/shares/boards/TQBR/securities/{ticker}/candles",
            "candles",
            {**params, "interval": 24},
        )
        frame["secid"] = ticker
        if not frame.empty:
            check_dates(frame, "begin", config["start"], config["end"])
        print(f"Candles {ticker}: {len(frame)} rows", flush=True)
        return frame

    with ThreadPoolExecutor(config["workers"]) as pool:
        candle_frames = list(pool.map(candles, tickers))
    empty_tickers = [
        ticker
        for ticker, frame in zip(tickers, candle_frames, strict=True)
        if frame.empty
    ]
    all_candles = pd.concat(candle_frames, ignore_index=True)
    diagnostic = []
    for ticker in config["history_tickers"]:
        frame = iss_pages(
            client,
            f"history/engines/stock/markets/shares/boards/TQBR/securities/{ticker}",
            "history",
            {"from": config["history_start"], "till": config["end"]},
        )
        if not frame.empty:
            check_dates(frame, "tradedate", config["history_start"], config["end"])
        diagnostic.append(frame)
        print(f"History {ticker}: {len(frame)} rows", flush=True)
    history = pd.concat(diagnostic, ignore_index=True)
    benchmark = iss_pages(
        client,
        "history/engines/stock/markets/index/securities/MCFTR",
        "history",
        params,
    )
    benchmark = benchmark.loc[
        (benchmark["boardid"] == "RTSI")
        & (benchmark["tradingsession"].astype(str) == "3")
    ].reset_index(drop=True)
    if benchmark.empty or benchmark["tradedate"].duplicated().any():
        raise ValueError("MCFTR benchmark does not provide a unique observed calendar")
    check_dates(benchmark, "tradedate", config["start"], config["end"])
    start = date.fromisoformat(config["start"]).strftime("%d/%m/%Y")
    end = date.fromisoformat(config["end"]).strftime("%d/%m/%Y")
    fx_url = "https://www.cbr.ru/scripts/XML_dynamic.asp?" + urlencode(
        {"date_req1": start, "date_req2": end, "VAL_NM_RQ": "R01235"}
    )
    fx = parse_fx(client.get(fx_url, "CBR USD/RUB"))
    key_url = "https://www.cbr.ru/hd_base/KeyRate/?" + urlencode(
        {
            "UniDbQuery.Posted": "True",
            "UniDbQuery.From": start.replace("/", "."),
            "UniDbQuery.To": end.replace("/", "."),
        }
    )
    key_rate = parse_key_rate(client.get(key_url, "CBR key rate"))
    for frame in [fx, key_rate]:
        check_dates(frame, "date", config["start"], config["end"])
    dataset_card_url = (
        f"https://huggingface.co/datasets/{config['news_dataset']}/raw/"
        f"{config['news_revision']}/README.md"
    )
    client.get(
        dataset_card_url, "RussianFinancialNews dataset card", config["news_revision"]
    )
    news_frames = []
    for split in ["train", "test"]:
        url = f"https://huggingface.co/datasets/{config['news_dataset']}/resolve/{config['news_revision']}/data/{split}-00000-of-00001.parquet"
        frame = pd.read_parquet(
            BytesIO(client.get(url, "RussianFinancialNews", config["news_revision"]))
        )
        frame["dataset_split"] = split
        news_frames.append(frame)
    news = pd.concat(news_frames, ignore_index=True)
    news, news_metadata = news_development_slice(news, config["start"], config["end"])
    original_split_summary = {}
    for split, frame in zip(["train", "test"], news_frames, strict=True):
        dates = pd.to_datetime(
            frame["date"], format="mixed", errors="coerce"
        ).dt.normalize()
        original_split_summary[split] = {
            "rows": len(frame),
            "date_min": dates.min().isoformat(),
            "date_max": dates.max().isoformat(),
            "unparseable_dates": int(dates.isna().sum()),
        }
    frames = {
        "candles": all_candles,
        "index_snapshots": snapshots,
        "history": history,
        "benchmark": benchmark,
        "fx": fx,
        "key_rate": key_rate,
        "news": news,
    }
    file_metadata = {}
    for name, frame in frames.items():
        target = processed / f"{name}.parquet"
        frame.to_parquet(target, index=False)
        file_metadata[name] = {
            "path": str(target.relative_to(root)),
            "rows": len(frame),
            "columns": list(frame.columns),
            "bytes": target.stat().st_size,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        }
    config_sha256 = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()
    source_records = sorted(client.records.values(), key=lambda record: record["url"])
    source_fingerprint = hashlib.sha256(
        json.dumps(
            [
                {key: record[key] for key in ["url", "sha256", "revision"]}
                for record in source_records
            ],
            sort_keys=True,
        ).encode()
    ).hexdigest()
    collection = {
        "config_sha256": config_sha256,
        "source_fingerprint": source_fingerprint,
        "config": config,
        "files": file_metadata,
        "tickers": tickers,
        "empty_candle_tickers": empty_tickers,
        **news_metadata,
        "original_split_summary": original_split_summary,
        "datasets_card_url": dataset_card_url,
    }
    write_json(processed / "collection.json", collection)
    manifest = {
        "config": config,
        "runtime": {
            "python": platform.python_version(),
            **{
                package: importlib.metadata.version(package)
                for package in [
                    "pandas",
                    "pyarrow",
                    "numpy",
                    "matplotlib",
                    "scipy",
                    "nbformat",
                    "nbclient",
                    "lxml",
                ]
            },
        },
        "config_sha256": config_sha256,
        "source_fingerprint": source_fingerprint,
        "sources": source_records,
    }
    write_json(root / "reports/eda/sources_manifest.json", manifest)
    print(
        json.dumps(
            {name: len(frame) for name, frame in frames.items()}, ensure_ascii=False
        ),
        flush=True,
    )
    return collection
