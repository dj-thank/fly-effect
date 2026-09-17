# Fly Effect — Circuit Bridge

ハエのコネクトームによる計算モデルと、将来の神経組織との入出力比較を目指す、独立研究系の出発点です。

**現在の成果は合成回路の比較ソフトウェアです。実コネクトーム、ハエの身体、ヒト細胞、生体組織を接続した成果ではありません。**

## GitHub上の位置づけ

- 上流: `dj-thank/fly-effect`
- 固定基準: `89d59fc56c3e4f3b0f9a26725ce4ee55fdf1ee36`
- 作業ブランチ: `research/circuit-bridge-fork-seed`
- 推奨する独立研究系名: `fly-effect-circuit-bridge`
- 現在の接続に新規リポジトリ / GitHub fork 作成操作がないため、フォーク移行用のブランチとして保存しています。**GitHub上の別フォークはまだ作成されていません。**
- 上流へのマージは目的にしません。元のファイルは変更せず、新しい研究用ファイルだけを追加します。未マージの他の研究PRも取り込んでいません。

## 実行

Python 3.11以降。実行時の追加依存はありません。リポジトリのルートで実行します。

```sh
git clone --branch research/circuit-bridge-fork-seed https://github.com/dj-thank/fly-effect.git fly-effect-circuit-bridge
cd fly-effect-circuit-bridge
python -m research.circuit_bridge --out work/circuit-bridge --steps 400 --seeds 0 1 2
```

`work/circuit-bridge/report.html` を開くと、条件とseedを切り替えて時系列を見られます。ネット接続やCDNは不要です。完全なデータは `result.json`、各試行の身体記録はCSVに保存します。既存の出力先は上書きしません。

```sh
python -m pip install 'pytest>=8,<10'
python -m pytest tests/circuit_bridge -q
```

これはソースツリーから使う研究ツールです。既存の `fly-effect` パッケージの配布内容・CLI・依存・全脳モデルを変更していません。

## 4つの比較条件

| 条件 | 処理 | 解釈 |
|---|---|---|
| intact | 合成回路をそのまま実行 | 比較の基準 |
| lesion | 指定した3つの内部ユニットを全ステップでゼロに固定 | 合成回路内の機能停止 |
| replacement | 同じ入出力境界に代替モデルを接続 | 既定は未学習の単純な合成モデル |
| shuffled | 指定領域の内部接続の行き先だけを並べ替え | 境界接続を保存した配線対照 |

合成身体は1次元の質量・減衰モデルで、ハエの6脚身体ではありません。ニューロンも無次元のrateモデルで、上流のBrian2 LIFではありません。異なるモデルを本物に見せかけません。

同一seed内では、全条件に同じ初期状態、時間刻み、目標系列、外部ノイズを与えます。一方、各条件の身体位置は異なり得るため、そこから戻る感覚入力も異なります。基準条件の感覚を固定再生する「偽の閉ループ」にはしていません。

内部接続のshuffleは接続本数、各source/weightの組、destinationの出現回数を保存します。重み付き入力総量や生物学的細胞型は保存する対照ではありません。多重辺と自己接続は許可します。変化しないshuffleは `shuffle_is_noop` に記録し、有効な対照と決めつけません。

## 差し替え自体が結果を変えないか

```sh
python -m research.circuit_bridge --out work/circuit-native --replacement native --steps 400 --seeds 7
```

`native` は元の内部回路と同じ式・配線を使うsham対照です。intactと同じ時系列になるかをテストしています。「代替回路が学習した」「生体組織が回復させた」という結果ではありません。

## 記録された入出力の再生

```sh
python -m research.circuit_bridge --out work/circuit-replay --replacement replay \
  --replay work/circuit-native/replacement-seed-7-replay.json --steps 400 --seeds 7
```

記録は合成由来であること、スキーマ、単位、チャンネル順、時刻、入力ハッシュ、記録全体のハッシュを持ちます。違う入力を与えたときに記録を流し続けることはせず、失敗で停止します。これはファイル再生であり、外部組織の応答性・学習・閉ループ適応を証明しません。

## 境界の契約

時刻tの全回路状態から境界への入力を計算し、代替回路はt+dtの領域状態を返します。領域外への出力は次の更新で使うため、直通による未来情報の漏れはありません。内部回路の代わりに領域全体を差し替え、外部への配線を維持します。

応答のステップ番号、dt、チャンネル順、無次元単位、入力SHA-256、形状、有限値、出力範囲を検証します。失敗した境界はラッチし、その試行は再利用しません。検証失敗時に回路・身体は次ステップへ進みません。生体向けの安全保証ではありません。

時間制限は同期関数が返った後に検出する**ソフト期限**です。停止しない外部関数を割り込んで止めることはできません。実機・生体への接続には、別プロセス、独立watchdog、装置固有の安全停止、単位校正、倫理審査が必要です。この版には実機の刺激コマンド、培養・移植手順はありません。

## 証拠の読み方

全グラフ、初期状態、外部刺激、結果のハッシュとseedを記録します。JSONの `claims` では実コネクトーム・上流脳実行・ハエ身体・生体接続・自律ハエ行動をすべてfalseとしています。

RMSEは合成課題内の記述統計です。回復率は `(lesion RMSE - replacement RMSE) / (lesion RMSE - intact RMSE)`。停止で悪化しない試行では計算せずnullにします。0〜1へクリップせず、負値や1超もそのまま記録します。複数seedは合成ノイズの反復であり、生物学的な個体数ではありません。p値・信頼区間・臨床効果は算出していません。

ローカル追加テストと上流全体のテスト、GitHub CI、科学的能力実証は別の検証です。追加テスト成功から元の全脳や6脚身体の正常動作を推定しません。

## 次の検証段階

1. **実装済み:** 合成小回路、停止・差し替え・shuffle、native対照、厳密ファイル再生、記録と表示。
2. **未実装:** 実コネクトームのID指定で対象領域と境界辺を抽出し、上流のgraph lockを通したBrian2モデルで同じ対照を実行。単位・シナプス遅延・refractory・checkpoint整合が必要。
3. **未実装:** 上流と同じ筋肉・身体・初期条件を接続。支持・推進校正を先に完了し、元の受入基準を弱めず比較。
4. **未実装:** 適切に取得された実記録の単位付きアダプター。記録再生を生体への能動的接続と区別。
5. **未実装:** 研究機関との生体接続。装置検証と倫理・安全監督を別途実施。ヒト細胞をハエ型身体につなぐことと、ハエ脳再現を混同しない。

## 研究との関係・出典

- Kaganovsky et al., *Developmental xenocortication using human-derived organoids in mice*, Nature (2026), DOI: `10.1038/s41586-026-11032-2`。著者機関の掲載: https://med.stanford.edu/pascalab/publications.html 。着想の背景であり、このソフトウェアで移植実験や指定配線の形成を再現したものではありません。
- Cai et al., *Brain organoid reservoir computing for artificial intelligence*, Nature Electronics 6, 1032–1039 (2023), DOI: `10.1038/s41928-023-01069-w`。https://www.nature.com/articles/s41928-023-01069-w 。多電極を介した入出力の先例であり、本実装はその装置・オルガノイドを使用していません。
- Fly Effectの原方針・未検証事項は上流 `README.md`、`docs/STATUS.md`、`organism_core/brain.py` を参照。既存の研究成果をこの合成デモの成果として転用しません。

## 独立リポジトリへの移行

このブランチには元のGit履歴が保持されます。別の移行先が作成された後、cloneした作業コピーで `origin` を `upstream` に変更し、新しい移行先を `origin` に登録して、この研究ブランチだけをpushできます。GitHubでのfork関係を持つリポジトリと、履歴をコピーした独立リポジトリは別物です。現在は前者の作成完了を主張しません。

新規コードは上流のMITライセンスに従います。外部研究・データ・図版を転載していません。
