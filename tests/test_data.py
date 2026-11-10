import json
from io import BytesIO
from unittest.mock import patch

import pytest

from moex_portfolio.data import (
    CachedHTTP,
    check_dates,
    index_snapshots,
    iss_pages,
    news_development_slice,
    parse_fx,
    parse_key_rate,
    validate_config,
)


def test_offline_miss_never_connects(tmp_path):
    with patch("moex_portfolio.data.urlopen") as request:
        with pytest.raises(FileNotFoundError):
            CachedHTTP(tmp_path, offline=True).get("https://example.com/data", "test")
        request.assert_not_called()


def test_cache_roundtrip_reuses_verified_payload(tmp_path):
    client = CachedHTTP(tmp_path / "data/raw")
    with patch(
        "moex_portfolio.data.urlopen", return_value=BytesIO(b"response")
    ) as request:
        assert client.get("https://example.com/data", "test") == b"response"
        assert client.get("https://example.com/data", "test") == b"response"
        assert request.call_count == 1
    with patch("moex_portfolio.data.urlopen") as request:
        assert (
            CachedHTTP(tmp_path / "data/raw", offline=True).get(
                "https://example.com/data", "test"
            )
            == b"response"
        )
        request.assert_not_called()


def test_attempts_must_be_positive(tmp_path):
    with pytest.raises(ValueError, match="attempts"):
        CachedHTTP(tmp_path, attempts=0)


def test_corrupt_cache_is_rejected(tmp_path):
    import hashlib

    url = "https://example.com/data"
    key = hashlib.sha256(url.encode()).hexdigest()
    (tmp_path / f"{key}.bin").write_bytes(b"changed")
    (tmp_path / f"{key}.json").write_text(
        json.dumps({"url": url, "sha256": "wrong", "bytes": 7, "revision": None})
    )
    with pytest.raises(ValueError, match="integrity"):
        CachedHTTP(tmp_path, offline=True).get(url, "test")


class FakeClient:
    def __init__(self, pages):
        self.pages = iter(pages)
        self.urls = []

    def get(self, url, source):
        self.urls.append(url)
        return json.dumps(next(self.pages)).encode()


def page(rows, index=0, total=3):
    return {
        "history": {"columns": ["SECID"], "data": [[x] for x in rows]},
        "history.cursor": {
            "columns": ["INDEX", "TOTAL", "PAGESIZE"],
            "data": [[index, total, 2]],
        },
    }


def test_cursor_pagination_uses_returned_count():
    client = FakeClient([page(["A", "B"]), page(["C"], 2)])
    result = iss_pages(client, "path", "history", {})
    assert result["secid"].tolist() == ["A", "B", "C"]
    assert "start=2" in client.urls[-1]


def test_truncated_cursor_is_an_error():
    client = FakeClient([page(["A", "B"]), page([], 2)])
    with pytest.raises(ValueError, match="Truncated"):
        iss_pages(client, "path", "history", {})


def test_repeated_page_is_an_error():
    client = FakeClient([page(["A"], total=2), page(["A"], 1, total=2)])
    with pytest.raises(ValueError, match="Repeated"):
        iss_pages(client, "path", "history", {})


def test_candles_continue_after_short_page():
    client = FakeClient(
        [
            {"candles": {"columns": ["begin"], "data": [["2024-01-01"]]}},
            {"candles": {"columns": ["begin"], "data": [["2024-01-02"]]}},
            {"candles": {"columns": ["begin"], "data": []}},
        ]
    )
    assert len(iss_pages(client, "path", "candles", {})) == 2
    assert len(client.urls) == 3


def test_repeated_candle_timestamp_inside_page_is_an_error():
    client = FakeClient(
        [{"candles": {"columns": ["begin"], "data": [["2024-01-01"], ["2024-01-01"]]}}]
    )
    with pytest.raises(ValueError, match="Unsorted"):
        iss_pages(client, "path", "candles", {})


def test_fx_nominal_is_applied():
    frame = parse_fx(
        b'<ValCurs><Record Date="01.01.2024"><Nominal>10</Nominal><Value>900,50</Value></Record></ValCurs>'
    )
    assert frame.iloc[0]["usd_rub"] == pytest.approx(90.05)


def test_key_rate_localized_table():
    payload = "<table><tr><th>Дата</th><th>Ставка</th></tr><tr><td>02.01.2024</td><td>16,00</td></tr><tr><td>01.01.2024</td><td>15,50</td></tr></table>".encode()
    frame = parse_key_rate(payload)
    assert frame["date"].tolist() == ["2024-01-01", "2024-01-02"]
    assert frame["rate"].tolist() == [15.5, 16.0]


def test_holdout_is_rejected():
    config = {
        "start": "2015-01-01",
        "end": "2025-01-01",
        "history_start": "2024-01-01",
        "holdout_start": "2026-01-01",
        "news_revision": "f" * 40,
        "workers": 4,
    }
    with pytest.raises(ValueError, match="development"):
        validate_config(config)


def test_source_outside_development_is_rejected():
    import pandas as pd

    with pytest.raises(ValueError, match="range"):
        check_dates(
            pd.DataFrame({"date": ["2025-01-01"]}), "date", "2015-01-01", "2024-12-31"
        )


def test_snapshot_holiday_fallback_preserves_requested_date():
    import pandas as pd

    config = {"start": "2024-01-01", "end": "2024-03-31", "workers": 1}
    frame = pd.DataFrame({"tradedate": ["2024-03-29"], "ticker": ["TEST"]})
    with patch(
        "moex_portfolio.data.iss_pages",
        side_effect=[pd.DataFrame(), pd.DataFrame(), frame],
    ) as fetch:
        result = index_snapshots(None, config)
    assert result["requested_date"].tolist() == ["2024-03-31"]
    assert result["tradedate"].tolist() == ["2024-03-29"]
    assert fetch.call_args.args[-1] == {"date": "2024-03-29"}


def test_snapshot_future_response_is_rejected():
    import pandas as pd

    config = {"start": "2024-01-01", "end": "2024-03-31", "workers": 1}
    frame = pd.DataFrame({"tradedate": ["2024-04-01"], "ticker": ["TEST"]})
    with patch("moex_portfolio.data.iss_pages", return_value=frame):
        with pytest.raises(ValueError, match="unexpected date"):
            index_snapshots(None, config)


def test_mixed_news_date_formats_obey_development_boundaries():
    import pandas as pd

    news = pd.DataFrame(
        {
            "date": [
                "2014-12-31 00:00:00",
                "2015-01-01",
                "2024-12-31 23:59:00",
                "2025-01-01",
            ],
            "title": ["old", "first", "last", "holdout"],
        }
    )
    filtered, metadata = news_development_slice(news, "2015-01-01", "2024-12-31")
    assert filtered["title"].tolist() == ["first", "last"]
    assert metadata["news_invalid_dates"] == 0
    assert metadata["news_rows_before_development"] == 1
    assert metadata["news_rows_after_development"] == 1
    assert metadata["news_original_rows"] == 4
