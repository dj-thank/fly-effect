"""Run with python -m research.circuit_bridge --out work/circuit-bridge."""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

from .core import BoundaryError, ReplayEndpoint, canonical
from .experiment import Config, run


def write_report(result: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    (output / "result.json").write_bytes(canonical(result) + b"\n")
    rows = []
    for trial in result["trials"]:
        name = f'{trial["condition"]}-seed-{trial["seed"]}'
        path = output / f"{name}.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            fields = [k for k in trial["trace"][0] if k != "rates"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({k: row[k] for k in fields} for row in trial["trace"])
        if trial["replacement_recording"] is not None:
            (output / f"{name}-replay.json").write_bytes(canonical(trial["replacement_recording"]) + b"\n")
        rows.append(f'<tr><td>{html.escape(trial["condition"])}</td><td>{trial["seed"]}</td>'
                    f'<td>{trial["rmse"]:.8f}</td><td><a href="{name}.csv">CSV</a></td></tr>')
    # Escape all '<' so JSON cannot terminate the inline script element.
    data = canonical(result).decode("utf-8").replace("<", "\\u003c")
    page = '''<!doctype html><html lang="ja"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fly Effect · Circuit Bridge</title><style>
body{margin:0;background:#101921;color:#edf5fa;font:16px/1.7 system-ui,sans-serif}
main{max-width:1060px;margin:auto;padding:32px 22px}h1{font-size:34px;margin-bottom:8px}
.label{color:#82d9d1;letter-spacing:.12em;font-size:13px}.notice{border-left:4px solid #efc06e;padding:12px 20px;background:#25313c}
.card{background:#1c2833;border:1px solid #3a4a58;border-radius:12px;padding:20px;margin-top:24px}
a{color:#a5e7e1}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:10px;border-bottom:1px solid #3a4a58}
select{padding:10px;background:#263744;color:white;border:1px solid #82d9d1;border-radius:5px}svg{width:100%;height:auto}
small{color:#becbd6}code{overflow-wrap:anywhere}h2{font-size:21px}.legend{color:#b3c9da}
</style><main><div class="label">FLY EFFECT / RESEARCH FORK SEED</div><h1>Circuit Bridge</h1>
<p>回路を止める・差し替える・配線を変える。同じ初期状態と外部刺激で比較。</p>
<div class="notice"><strong>合成回路のソフトウェア検証</strong><br>
この実行は実コネクトーム、ハエ身体、生体組織を使用していません。歩行・学習・意識の実証ではありません。</div>
<section class="card"><h2>閉ループの時系列</h2><select id="trial" aria-label="試行を選択"></select>
<p class="legend">水色：合成身体の位置（時刻 t の更新前）　黄色：同時刻の目標</p>
<svg id="chart" viewBox="0 0 940 320" role="img" aria-label="位置と目標の時系列"></svg>
<small>横軸：秒　縦軸：合成モデルの任意単位。データ点は result.json と CSV に保存。</small></section>
<section class="card"><h2>全試行の測定結果</h2><table><thead><tr><th>条件</th><th>Seed</th><th>位置追従RMSE</th><th>記録</th></tr></thead><tbody>ROWS</tbody></table>
<p>RMSEはこの合成課題内の指標で、生物学的能力のスコアではありません。</p></section>
<section class="card"><h2>再現性</h2><p><a href="result.json">完全な結果・マニフェスト JSON</a></p>
<p>上流基準コミット：<code>BASE</code></p><p>結果SHA-256：<code>HASH</code></p>
<p>乱数Seed、グラフ、初期状態、外部ノイズ、目標系列、トレースのハッシュを記録。感覚フィードバックは各条件の身体から計算されます。</p></section>
<script id="data" type="application/json">DATA</script><script>
const data=JSON.parse(document.getElementById('data').textContent);const select=document.getElementById('trial');
data.trials.forEach((t,i)=>{const o=document.createElement('option');o.value=i;o.textContent=t.condition+' / seed '+t.seed;select.appendChild(o)});
function draw(){const t=data.trials[Number(select.value)].trace;const values=t.flatMap(r=>[r.position_before,r.target]);
const low=values.reduce((a,v)=>Math.min(a,v),0)-.05,high=values.reduce((a,v)=>Math.max(a,v),0)+.05,maxX=Math.max(t.at(-1).t_s,.01);
const x=v=>55+v/maxX*865,y=v=>270-(v-low)/(high-low)*245;
const pts=k=>t.map(r=>x(r.t_s).toFixed(2)+','+y(r[k]).toFixed(2)).join(' ');
let grid='';for(let i=0;i<5;i++){const v=low+(high-low)*i/4;grid+=`<line x1="55" x2="920" y1="${y(v)}" y2="${y(v)}" stroke="#364755"/><text x="0" y="${y(v)+5}" fill="#bdceda" font-size="13">${v.toFixed(2)}</text>`}
document.getElementById('chart').innerHTML=grid+`<polyline fill="none" stroke="#edc078" stroke-width="2" points="${pts('target')}"/><polyline fill="none" stroke="#83e0d5" stroke-width="2" points="${pts('position_before')}"/><text x="55" y="307" fill="#bdceda">0 s</text><text x="845" y="307" fill="#bdceda">${maxX.toFixed(2)} s</text>`;}
select.addEventListener('change',draw);draw();</script></main></html>'''
    page = page.replace("ROWS", "".join(rows)).replace("BASE", result["upstream_base_commit"])
    page = page.replace("HASH", result["result_sha256"]).replace("DATA", data)
    (output / "report.html").write_text(page, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Synthetic circuit replacement comparison, not a biological fly simulation")
    parser.add_argument("--out", type=Path, required=True, help="New directory; existing results are never overwritten")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--replacement", choices=["surrogate", "native", "replay"], default="surrogate")
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.out.exists():
            raise ValueError("Output already exists; use a new path to preserve evidence")
        if (args.replacement == "replay") != (args.replay is not None):
            raise ValueError("--replay is required only with --replacement replay")
        if args.replay is not None and len(args.seeds) != 1:
            raise ValueError("Exact replay requires one seed matched to the recording")
        config = Config(dt_s=args.dt, steps=args.steps, seeds=tuple(args.seeds), replacement=args.replacement)
        factory = (lambda: ReplayEndpoint.from_file(args.replay)) if args.replay else None
        result = run(config, endpoint_factory=factory)
        write_report(result, args.out)
    except (ValueError, OSError, BoundaryError, TypeError, KeyError) as exc:
        parser.exit(2, f"Circuit Bridge stopped: {exc}\n")
    print(json.dumps({"status": "completed", "evidence": result["evidence"],
                      "trials": len(result["trials"]), "result_sha256": result["result_sha256"],
                      "report": str(args.out / "report.html")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
