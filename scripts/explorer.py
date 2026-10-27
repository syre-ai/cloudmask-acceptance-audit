"""A small interactive view for the site: drag OmniCloudMask's P(clear) threshold and watch leak
and discard move along the measured trade-off curve, next to Cloud Score+ and s2cloudless.
All values come from results/curves_test.csv and results/operating_points_test.csv.

    uv run python scripts/explorer.py   -> site/explorer/index.html (self-contained)
"""

import json
from pathlib import Path

import pandas as pd

CURVES = {
    "OmniCloudMask v4 | l1c": "OmniCloudMask v4",
    "Cloud Score+ cs | product": "Cloud Score+ (cs)",
    "s2cloudless raw probability | product": "s2cloudless (raw probability)",
}

TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cloud-mask threshold explorer</title>
<style>
  :root { --surface:#ffffff; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7;
          --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; }
  @media (prefers-color-scheme: dark) { :root:where(:not([data-theme="light"])) {
          --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --muted:#898781; --grid:#2c2c2a; --axis:#383835;
          --s1:#3987e5; --s2:#d95926; --s3:#199e70; } }
  :root[data-theme="dark"] { --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --muted:#898781; --grid:#2c2c2a;
          --axis:#383835; --s1:#3987e5; --s2:#d95926; --s3:#199e70; }
  body { margin:0; padding:16px; background:var(--surface); color:var(--ink);
         font:14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
  .wrap { max-width:760px; }
  h1 { font-size:17px; margin:0 0 4px; }
  p { color:var(--ink2); margin:0 0 12px; }
  .tiles { display:flex; gap:12px; flex-wrap:wrap; margin:10px 0 6px; }
  .tile { border:1px solid var(--grid); border-radius:8px; padding:8px 12px; min-width:140px; }
  .tile .l { color:var(--ink2); font-size:12px; }
  .tile .v { font-size:22px; font-weight:600; font-variant-numeric:tabular-nums; }
  input[type=range] { width:100%; }
  svg text { fill:var(--ink2); font-size:11px; }
  .legend { display:flex; gap:16px; flex-wrap:wrap; color:var(--ink2); font-size:12px; margin-top:6px; }
  .key { display:inline-block; width:16px; height:2px; vertical-align:middle; margin-right:6px; }
  .note { color:var(--muted); font-size:12px; margin-top:8px; }
</style>
</head>
<body>
<div class="wrap">
<h1>What a cloud-mask threshold costs you</h1>
<p>Move the slider to change how confident OmniCloudMask must be before a pixel is accepted as clear. <b>Leak</b>: accepted pixels that are actually cloud or shadow. <b>Discard</b>: clear pixels thrown away. Measured on __N__ expert-labelled Sentinel-2 test scenes.</p>
<label for="t" style="color:var(--ink2)">Accept a pixel if P(clear) is at least <b id="tv"></b></label>
<input id="t" type="range" min="0" max="__MAXI__" step="1">
<div class="tiles">
  <div class="tile"><div class="l">Leak</div><div class="v" id="leak"></div></div>
  <div class="tile"><div class="l">Discard</div><div class="v" id="disc"></div></div>
  <div class="tile"><div class="l">Recommended for 2% leak</div><div class="v" id="rec" style="font-size:15px"></div></div>
</div>
<svg id="chart" viewBox="0 0 720 400" role="img" aria-label="Leak against discard trade-off curves"></svg>
<div class="legend" id="legend"></div>
<p class="note">Curves sweep each product's own threshold on the CloudSEN12+ test split (overlap-screened). The recommended point was chosen on the validation split. Source: cloudmask-acceptance-audit (results/curves_test.csv).</p>
</div>
<script>
const D = __DATA__;
const svg = document.getElementById("chart"), NS = "http://www.w3.org/2000/svg";
const W = 720, H = 400, m = {l:74, r:16, t:12, b:44};
const XMAX = 0.6, YMAX = 0.35;
const x = (v) => m.l + Math.min(v, XMAX) / XMAX * (W - m.l - m.r);
const y = (v) => H - m.b - Math.min(v, YMAX) / YMAX * (H - m.t - m.b);
const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
function el(tag, attrs, parent = svg) { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); parent.appendChild(e); return e; }
const pct = (v) => (100 * v).toFixed(1) + "%";
function draw() {
  svg.innerHTML = "";
  for (let i = 0; i <= 6; i++) { const v = i * 0.1; el("line", {x1:x(v), x2:x(v), y1:y(0), y2:y(YMAX), stroke:css("--grid")}); el("text", {x:x(v), y:H - m.b + 16, "text-anchor":"middle"}).textContent = Math.round(v * 100) + "%"; }
  for (let i = 0; i <= 7; i++) { const v = i * 0.05; el("line", {x1:x(0), x2:x(XMAX), y1:y(v), y2:y(v), stroke:css("--grid")}); el("text", {x:m.l - 8, y:y(v) + 4, "text-anchor":"end"}).textContent = Math.round(v * 100) + "%"; }
  el("text", {x:(W + m.l) / 2, y:H - 6, "text-anchor":"middle"}).textContent = "Clear pixels thrown away (discard)";
  el("text", {x:14, y:(H - m.b) / 2, "text-anchor":"middle", transform:`rotate(-90 14 ${(H - m.b) / 2})`}).textContent = "Leak: cloud or shadow among accepted";
  D.series.forEach((s, k) => {
    const pts = s.points.filter((p) => p[1] <= XMAX && p[2] <= YMAX);
    el("polyline", {points: pts.map((p) => `${x(p[1])},${y(p[2])}`).join(" "), fill:"none", stroke:css(`--s${k + 1}`), "stroke-width":2, "stroke-linejoin":"round"});
  });
  const r = D.rec; el("circle", {cx:x(r.discard), cy:y(r.leak), r:5, fill:"none", stroke:css("--ink"), "stroke-width":1.5});
  const i = +document.getElementById("t").value, p = D.series[0].points[i];
  el("circle", {cx:x(p[1]), cy:y(p[2]), r:6, fill:css("--s1"), stroke:css("--surface"), "stroke-width":2});
  document.getElementById("tv").textContent = p[0].toFixed(3);
  document.getElementById("leak").textContent = pct(p[2]);
  document.getElementById("disc").textContent = pct(p[1]);
}
document.getElementById("legend").innerHTML = D.series.map((s, k) => `<span><span class="key" style="background:var(--s${k + 1})"></span>${s.name}</span>`).join("")
  + `<span><svg width="14" height="14" style="vertical-align:middle;margin-right:6px"><circle cx="7" cy="7" r="5" fill="none" stroke="currentColor" stroke-width="1.5"/></svg>recommended operating point (validation-chosen)</span>`;
document.getElementById("rec").textContent = `${D.rec.rule}: leak ${pct(D.rec.leak)}, discard ${pct(D.rec.discard)}`;
const slider = document.getElementById("t"); slider.value = D.recIndex; slider.oninput = draw;
if (window.matchMedia) matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);
draw();
</script>
</body>
</html>
"""


def main() -> None:
    cv = pd.read_csv("results/curves_test.csv")
    ops = pd.read_csv("results/operating_points_test.csv")
    ov = pd.read_csv("results/overall_test.csv").set_index("group")
    series = []
    for key, name in CURVES.items():
        d = cv[cv["method"] == key].dropna()
        pts = d[["threshold_native", "discard_rate", "leak_rate"]].to_numpy()
        if key.startswith("OmniCloudMask"):
            # curve file stores t/255; the executable rule for quantized level t is P >= (t - 0.5)/255
            t = (pts[:, 0] * 255).round()
            pts[:, 0] = ((t - 0.5).clip(min=0)) / 255
        series.append({"name": name, "points": pts.round(5).tolist()})
    r = ops[(ops["method"] == "OmniCloudMask v4 | l1c") & ops["rule"].str.startswith("leak <= 2%")].iloc[0]
    rec = {"rule": f"P(clear) at least {r['threshold_native']:.3f}", "leak": float(r["test_leak"]),
           "discard": float(r["test_discard"])}
    ocm = series[0]["points"]
    rec_index = min(range(len(ocm)), key=lambda i: abs(ocm[i][0] - r["threshold_native"]))
    data = {"series": series, "rec": rec, "recIndex": rec_index}
    html = (TEMPLATE.replace("__DATA__", json.dumps(data))
            .replace("__MAXI__", str(len(ocm) - 1))
            .replace("__N__", str(int(ov.loc["OmniCloudMask v4 | l1c", "images"]))))
    if "—" in html:
        raise SystemExit("em dash in explorer")
    out = Path("site/explorer")
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(html)
    print(f"site/explorer/index.html ({len(html) // 1024} KB)")


if __name__ == "__main__":
    main()
