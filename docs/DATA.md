# データと再配布

生物データ、第三者メッシュ、学習済み重み、ローカル実行記録は配布しません。コードのライセンスがデータの権利を変更することはありません。取得元の条件と引用指定を確認してください。

## 現アダプターの形式

FLY_EFFECT_DATA_DIR/graphに次の処理済みファイルを置きます。

- body_ids.npy: 元ニューロンID配列
- motor_indices.npy: 815運動ニューロンの内部インデックス
- glutamate_inhibitory_hypothesis_signs.npy: -1/0/1の明示的符号仮説
- neurons.parquet: 実ID、型、神経、側などの注釈
- edges.bin: little-endian。pre:uint32, post:uint32, count:uint32, source_row:uint64

固定ハッシュはorganism_core/graph_lock.jsonにあります。166,700ニューロンと25,582,938接続行を保持します。このカスタム形式の構築手順を公開可能な形にすることは未完了の優先課題です。ハッシュがあるだけで誰でも取得できるとは主張しません。

筋対応にはDATA_DIR/manc-supplements/elife-96084-supp3-v1.csvと、ローカル生成する対応仮説が必要です。外部素材を追加する提案では出典・版・ライセンス・加工内容・ハッシュを示し、権利確認前にバイナリをコミットしないでください。

参考: [FlyEM](https://www.janelia.org/project-team/flyem)、[FlyGym](https://github.com/NeLy-EPFL/flygym)、[CPG研究](https://github.com/smpuglie/Pugliese_cpg_2025)。各提供元は独立しており、リンクは再配布許可を意味しません。


## 接続率と運動出力の監査

登録数、非運動ニューロンの接続率、非ゼロ重み、運動系への経路、筋肉側への割当を区別する[読み取り専用監査](CONNECTIVITY_AUDIT.md)を利用できます。配線の追加・削除や受入条件の緩和は行いません。
