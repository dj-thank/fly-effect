# Virtual Fly Brain データを Fly Effect へ取り込む

この実装は VFB の知識グラフを別保存し、既存の実 ID と照合する入口です。
**「取り込みコードがある」「API から取れた」「モデルが利用した」「生物と一致した」は別です。**
全画像・全コネクトームの取得済み、安定立位・歩行・個体クローンの完成は意味しません。

## データの使い分け

| 情報 | 取得・保持するもの | 利用先と境界 |
|---|---|---|
| EM 接続 | 全ノードと関係、元 ID、符号とは独立した count、出典 | 配線の照合・追加候補。現行 graph を自動置換しない |
| 神経形態・標準脳 | channel、depicts、in_register_with、登録空間、配布 URL | SWC/OBJ/NRRD/Woolz の取得候補。座標変換や画像本体取得は別工程 |
| 細胞型・領域・発生 | Class/Individual、型、階層、時期、性、左右、Deprecated | 同一種でも個体・性・齢・版を区別して注釈照合 |
| 運動・感覚器 | 神経クラスと筋肉・感覚器の関係と出典 | 身体接続の候補。個別運動ニューロンの実証済み割当ではない |
| 遺伝子・scRNA-seq | gene、cluster、expresses 等の全プロパティ | 分子型の制約。発現から膜電位・コンダクタンスを捏造しない |
| LM・発現・クローン | 染色、driver、形態比較、画像登録の注釈 | 解剖学的照合。活動時系列とは区別 |
| 文献・ライセンス | DataSet、Site、License、pub と関連プロパティ | 出典・権利・版の追跡。ライセンス不明は再配布許可でない |

VFB の統合先は PDB です。`all` は**実行時の PDB の全ノード・全関係**を対象とし、
未知のラベルやプロパティも捨てません。5シナプスの既定閾値、特定 EM データセットの
既定除外を適用しません。`Related` と名前付き関係の重複表現は元のまま保存し、
両方を別の生物学的シナプスとして足しません。取得数は PDB の graph record 数です。

PDB に統合されていない CATMAID 専用の ABD1.5、IAV-ROBO、IAV-TNT、L3VNC、
Larva1099 等、外部の原画像・生データ、VFB が収録していない遺伝子発現は別です。
その未実装コネクターを取得済みとしません。VFB scRNA-seq は発現 extent > 0.2 の
選別済み情報なので、欠落を発現ゼロとしません。変異・操作個体を野生型と混ぜません。

## 実行

Python 3.11 以上。取得は標準ライブラリだけで動作し、GPU や新しい有料サービスは不要です。
外部サービスへのアクセス許可とネットワークは必要です。並列 bulk 要求はせず、提供元へ
負荷をかけない範囲の予算で実行してください。CI の自動確認は小さい2範囲だけです。

```bash
# DataSet/Site/License/Template とその出方向の関係。全データセットを動的に列挙。
python -m organism_core.vfb_data export --scope catalogue --out work/vfb-catalogue
python -m organism_core.vfb_data verify work/vfb-catalogue --summary work/catalogue-summary.json

# 神経クラス→筋肉/感覚器の関係。関係名・出典と両端の注釈を保持。
python -m organism_core.vfb_data export --scope body --out work/vfb-body

# 元グラフの版に対応するソースを明示。旧版や別個体へ自動フォールバックしない。
python -m organism_core.vfb_data export --scope xrefs --source-db male_cns_v1_0 \
  --out work/vfb-male-xrefs --max-pages 500 --max-rows 1000000 --wall-seconds 600

# 原データの body_ids.npy が必要。固定 SHA256 が一致しなければ拒否。
python -m organism_core.vfb_enrichment --xrefs work/vfb-male-xrefs \
  --body-ids /path/to/graph/body_ids.npy --out work/vfb-neuron-evidence

# 全 PDB メタデータ。大規模処理であり、この PR の CI では実行しない。
python -m organism_core.vfb_data export --scope all --out work/vfb-all \
  --page-size 500 --max-pages 100 --max-rows 50000 --wall-seconds 600
# 予算切れ (exit 3) の続き。取得済みページを検算後、同一条件で再開。
python -m organism_core.vfb_data export --scope all --out work/vfb-all --resume \
  --page-size 500 --max-pages 100 --max-rows 50000 --wall-seconds 600
```

取得済みの body_ids.npy や実全脳データは Git に含まれません。公開 API と現行カスタム
グラフの母集団は異なり得ます。一致割合は、実際に渡した同じ ID 母集団でだけ計算します。
`matched/unmatched/ambiguous/excluded` を別記し、廃止済み・非神経個体・多対一を
勝手に統合しません。未取得・不完全な xrefs から完成率を出すことは拒否します。

## 出力と失敗

取得先は未使用ディレクトリ必須です。各 JSONL ページに SHA256、行数、末尾の
内部レコード ID を付け、receipt に scope、endpoint、source_db、取得前後件数、
DataSet 内容のハッシュ、query ハッシュと予算を残します。
未知フィールド、元関係、外部アクセッションは保持します。内部 `id(n)` / `id(r)` は
**その PDB 内でのページング用**で、生物 ID ではありません。

`complete_count_checked` は選択範囲の列挙件数の整合だけです。ライブ DB のページ取得は
単一トランザクションの固定スナップショットではなく、件数が同じ更新を検出できない
場合があります。`snapshot_is_atomic=false` を常に残します。再開時に件数・DataSet・
query が変わった場合は新しい作業先でやり直します。論文向けの固定完全グラフには、
提供元と調整した immutable dump/版の確定が別途必要です。

成功は exit 0、API・形式・ハッシュ・不整合は exit 2、取得予算切れは exit 3 です。
失敗を生物学的不合格や正常な空データへ変換しません。`verify` は全保存ページを検算し、
不完全な取得を成功扱いしません。新規実行による既存データの上書きは拒否します。
異常終了で未登録のページだけ残った場合、自動で上書き/削除しません。証拠を保全して
別作業先から再実行するか、未登録ファイルを手動で隔離してください。

ページ/行/bytes 予算は1回の呼出しごとです。壁時計予算は処理間の協調的検査、HTTP
はソケット timeout であり、OS 強制停止・サーバーの query cancellation・RSS hard cap
ではありません。`all` のライブ走査性能は全規模で未検証です。長い処理は外側でも予算を
設定し、提供元と bulk 利用を調整してください。ローカル workspace は信頼できる場所で
**1 writer**だけ使います。ハッシュは出典の署名や悪意ある全書換えの防止ではありません。

`neuron-evidence.jsonl` は導出した注釈 sidecar です。既存 Engine の重み、感覚入力、
運動割当、筋腱、graph lock を変更しません。実効的なモデル適用には各注釈の対応確認と
対照実験が必要です。古い checkpoint の code_hash を書き換えて強制復元しないでください。
新モジュール追加で既存 code identity が変わり得ます。

## 取得範囲を過大評価しない

- `catalogue`: カタログと出方向 metadata の取得。全画像本体ではない。
- `body`: クラス間関係の取得。シナプス数・筋力・実ニューロン割当ではない。
- `xrefs`: 版を指定した ID と元注釈。配線・活動・画像本体ではない。
- `all`: PDB にある全 graph record の取得経路。現時点で全件取得済みではない。
- 原画像/メッシュ取得、座標変換、外部 CATMAID 原データ、既存神経計算への適用は未完。

CI には外部原データを publish せず、件数・関係別集計・page hashes の summary のみを
保存します。raw データは work/ に隔離し、原提供元の条件・引用を確認してから扱います。
最新の実接続/CI の結果は PR の検証記録で確認し、親 PR の緑を流用しません。

## 一次資料

- [VFB データ一覧](https://www.virtualflybrain.org/docs/data/)
- [PDB のラベル・関係・画像構造](https://www.virtualflybrain.org/docs/apis/pdb/)
- [接続情報とクラス間関係](https://www.virtualflybrain.org/docs/data/connectivity/)
- [EM データと PDB 非統合データ](https://www.virtualflybrain.org/docs/data/em/)
- [scRNA-seq の収録条件](https://www.virtualflybrain.org/docs/data/scrnaseq/)
- [実行時の収録レポート](https://www.virtualflybrain.org/blog/2022/01/01/vfb-content-report/)
- [VFB_connect の公開 endpoint と credentials](https://github.com/VirtualFlyBrain/VFB_connect/blob/master/src/vfb_connect/default_servers.py)
- [公式 ID 変換の実装](https://github.com/VirtualFlyBrain/VFB_connect/blob/master/src/vfb_connect/neo/query_wrapper.py)

2026-09-12確認時、EM 概要の male-CNS v0.9/BANC v626 と収録レポートの
male_cns_v1_0/BANC888 に版の差がありました。固定の一覧を取得済みとせず、実取得時の
カタログと endpoint に基づいて確認します。VFB 全体の件数を現行モデルの分母に流用しません。
