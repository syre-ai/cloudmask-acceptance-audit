import io

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio as rio

from cmaudit.download import BAND_INDEX, decode_level, parse_tortilla


def _tif_bytes(arr: np.ndarray) -> bytes:
    arr = arr if arr.ndim == 3 else arr[None]
    buf = io.BytesIO()
    with rio.MemoryFile() as mf:
        with mf.open(driver="GTiff", count=arr.shape[0], height=arr.shape[1],
                     width=arr.shape[2], dtype=arr.dtype) as dst:
            dst.write(arr)
        buf.write(mf.read())
    return buf.getvalue()


def _tortilla(children: dict[str, bytes]) -> bytes:
    """Build a container the way CloudSEN12+ stores samples: header, data, parquet footer."""
    body, rows, pos = b"", [], 18
    for tid, data in children.items():
        rows.append({"tortilla:id": tid, "tortilla:offset": pos, "tortilla:length": len(data)})
        body += data
        pos += len(data)
    sink = io.BytesIO()
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(rows)), sink)
    footer = sink.getvalue()
    header = b"#y" + pos.to_bytes(8, "little") + len(footer).to_bytes(8, "little")
    return header + body + footer


def test_parse_and_decode_level_sample():
    img = np.arange(13 * 8 * 8, dtype=np.uint16).reshape(13, 8, 8)
    lab = (np.arange(64, dtype=np.uint8) % 4).reshape(8, 8)
    blob = _tortilla({"s2l1c": _tif_bytes(img), "target": _tif_bytes(lab)})
    kids = parse_tortilla(blob)
    assert set(kids) == {"s2l1c", "target"}
    out = decode_level(kids)
    # stored bands are B02, B03, B04, B8A = 1-based 2, 3, 4, 9
    for k, (name, band) in enumerate(BAND_INDEX.items()):
        assert np.array_equal(out["bands"][k], img[band - 1]), name
    assert np.array_equal(out["label"], lab)
