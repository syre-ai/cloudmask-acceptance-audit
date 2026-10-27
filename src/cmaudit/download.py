"""Fetch CloudSEN12+ evaluation samples with one HTTP range request each.

The TACO index gives every sample's byte range inside a ~20 GB shard. Each sample is itself
a small "tortilla" container (18-byte header, child files, parquet footer listing the
children's offsets relative to the sample start), so one ranged GET retrieves everything and
the children are parsed in memory. This keeps the request count low enough for anonymous
Hugging Face rate limits (a 429 makes every worker back off).

Each sample is stored as ``data/samples/<level>/<tortilla:id>.npz`` with:
- ``bands``: uint16 (4, 512, 512) = B02, B03, B04, B8A (blue, green, red, NIR narrow)
- ``label``: uint8 (512, 512) expert label (0 clear, 1 thick, 2 thin, 3 shadow)
and ``data/samples/extra/<tortilla:id>.npz`` with the operational baseline masks, ESA
WorldCover and elevation, each (512, 512).
"""

from __future__ import annotations

import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests
from pyarrow import BufferReader
from rasterio.io import MemoryFile
from tqdm import tqdm

# 1-based band indexes in the 13-band CloudSEN12 GeoTIFF (B01..B12 incl. B8A and B10).
BAND_INDEX = {"B02": 2, "B03": 3, "B04": 4, "B8A": 9}
BANDS = list(BAND_INDEX)

EXTRA_LAYERS = [
    "elevation", "lc10", "cloudmask_qa60", "cloudmask_sen2cor", "cloudmask_s2cloudless",
    "cloudmask_cloudscore_cs_v1", "cloudmask_cloudscore_cs_cdf_v1", "cloudmask_unetmobv2_v1",
    "cloudmask_unetmobv2_v2", "cloudmask_sensei_v2",
]

_SUBFILE = re.compile(r"/vsisubfile/(\d+)_(\d+),/vsicurl/(.+)")


class _Backoff:
    """Shared pause so all workers stop when the server says 429."""

    def __init__(self) -> None:
        self.until = 0.0
        self.lock = threading.Lock()

    def wait(self) -> None:
        while (d := self.until - time.time()) > 0:
            time.sleep(min(d, 5))

    def hit(self, seconds: float) -> None:
        with self.lock:
            self.until = max(self.until, time.time() + seconds)


_backoff = _Backoff()


class _Resolver:
    """Cache each shard's signed CDN redirect target.

    Hugging Face rate-limits its /resolve/ endpoint (3000 requests / 5 min anonymously), but
    range reads against the CDN URL it redirects to are not counted there. Resolve once per
    shard and refresh every 15 minutes or when the signed URL stops working.
    """

    TTL = 15 * 60

    def __init__(self) -> None:
        self.cache: dict[str, tuple[str, float]] = {}
        self.lock = threading.Lock()

    def get(self, url: str, refresh: bool = False) -> str:
        with self.lock:
            hit = self.cache.get(url)
            if hit and not refresh and time.time() - hit[1] < self.TTL:
                return hit[0]
            _backoff.wait()
            r = requests.head(url, allow_redirects=False, timeout=60)
            if r.status_code == 429:
                _backoff.hit(float(r.headers.get("Retry-After", 60)))
                raise RuntimeError("rate limited while resolving")
            target = r.headers.get("Location", url) if r.is_redirect else url
            self.cache[url] = (target, time.time())
            return target


_resolver = _Resolver()


def fetch_range(url: str, offset: int, length: int, tries: int = 8) -> bytes:
    refresh = False
    for attempt in range(tries):
        _backoff.wait()
        try:
            cdn = _resolver.get(url, refresh=refresh)
            r = requests.get(
                cdn, headers={"Range": f"bytes={offset}-{offset + length - 1}"}, timeout=120
            )
        except (requests.RequestException, RuntimeError):
            time.sleep(2**attempt)
            continue
        if r.status_code == 206 and len(r.content) == length:
            return r.content
        if r.status_code == 429:
            _backoff.hit(float(r.headers.get("Retry-After", 60 * (attempt + 1))))
            continue
        refresh = r.status_code in (400, 401, 403)  # signed URL expired
        time.sleep(2**attempt)
    raise RuntimeError(f"range fetch failed after {tries} tries: {url} @{offset}")


def parse_tortilla(blob: bytes) -> dict[str, bytes]:
    """Split a tortilla container into {child id: child bytes}."""
    if blob[:2] not in (b"#y", b"WX"):
        raise ValueError("not a tortilla container")
    fo = int.from_bytes(blob[2:10], "little")
    fl = int.from_bytes(blob[10:18], "little")
    footer = pq.read_table(BufferReader(blob[fo : fo + fl])).to_pandas()
    return {
        tid: blob[int(o) : int(o) + int(n)]
        for tid, o, n in zip(footer["tortilla:id"], footer["tortilla:offset"],
                             footer["tortilla:length"])
    }


def _tif(b: bytes, bands=None) -> np.ndarray:
    with MemoryFile(b) as mf, mf.open() as src:
        return src.read(bands) if bands is not None else src.read(1)


def decode_level(children: dict[str, bytes]) -> dict:
    img_key = next(k for k in children if k != "target")
    bands = _tif(children[img_key], [BAND_INDEX[b] for b in BANDS])
    return {"bands": bands.astype(np.uint16), "label": _tif(children["target"]).astype(np.uint8)}


def decode_extra(children: dict[str, bytes]) -> dict:
    return {name: _tif(children[name]) for name in EXTRA_LAYERS}


def fetch(mode: str, index: pd.DataFrame, ids: list[str], out_dir: Path,
          workers: int = 4) -> list[str]:
    """Download the given sample ids for one TACO mode; returns ids that failed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = index.set_index("tortilla:id").loc[ids]
    todo = [t for t in ids if not (out_dir / f"{t}.npz").exists()]
    decode = decode_extra if mode == "extra" else decode_level

    def job(tid: str) -> str | None:
        sub = rows.loc[tid, "internal:subfile"]
        offset, length, url = _SUBFILE.match(sub).groups()
        try:
            arrays = decode(parse_tortilla(fetch_range(url, int(offset), int(length))))
        except Exception as exc:  # noqa: BLE001 - report and let the caller retry
            print(f"[{mode}] {tid}: {type(exc).__name__}: {exc}")
            return tid
        tmp = out_dir / f"{tid}.tmp.npz"
        np.savez_compressed(tmp, **arrays)
        tmp.rename(out_dir / f"{tid}.npz")
        return None

    failed = []
    with ThreadPoolExecutor(workers) as pool:
        futs = [pool.submit(job, t) for t in todo]
        for f in tqdm(as_completed(futs), total=len(futs), desc=mode, mininterval=10):
            if (r := f.result()) is not None:
                failed.append(r)
    return failed

