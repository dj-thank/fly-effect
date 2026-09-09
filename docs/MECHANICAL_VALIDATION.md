# 筋力・身体の切り分け検証

全CNSの配線、ニューロンID、シナプス重み、`graph_lock.json` は変更しません。
歩行の失敗を全脳の調整だけで解こうとせず、まず筋力計算と身体応答を独立に検証します。
ここでの刺激は**人工的な診断入力**です。歩行制御器でも、生物学的な自律運動の証拠でもありません。

## 1. 外部データ不要の力学検査

Python 3.11以上で、次を実行します。

```bash
python -m pip install -e ".[dev,neural,mechanics]"
fly-effect mechanics-demo --out work/mechanics-check
python -m pytest -q
```

自由基部と1つの回転関節を持つ、独自の解析用モデルを使います。ハエの形態や神経配線は使いません。
正の張力を F、腱の長さを L とすると、一般化力は `-dL/dq * F` です。
サイトへの作用・反作用から求めた力を、腱長の中央差分という独立の方法と比較します。
自由関節の姿勢はMuJoCoの `mj_integratePos` を使って摂動し、四元数の成分を直接足し引きしません。

正しい力は解析解と一致し、意図的に符号を反転した負の対照は不一致になることを検査します。
入力個数の不一致、負・非有限の張力、不正なサイトID、退化した経路を拒否します。
球や円柱への巻き付き、プーリー、固定腱は、この直線サイト用の計算に紛れ込ませず明示的に拒否します。
実験の成功と生物学的な再現は別です。`biological_validation` と `walking_claimed` は false のままです。

## 2. 同じ六脚身体への対照刺激

**固定したFlyGymは Python >=3.12,<3.15 を必要とします。**
コア・人工神経回路のPython 3.11対応とは別の条件です。
以下はPython 3.12の環境で実行してください。

```bash
python -m pip install -e ".[body]"
fly-effect doctor
fly-effect calibrate-body --out work/body-probe --duration 0.02 --leg lf
```

全脳のデータや運動ニューロン対応表は不要ですが、FlyGym本体と身体のメッシュは必要です。
外部素材は提供元のライセンスに従い、実行時の取得にはネットワークが必要になる場合があります。
`[body]` は身体だけの診断、`[simulation]` は追加の神経・注釈データを使う全系実験向けです。

同一の物理状態・筋肉状態から、無入力、正側のNMJへの1イベント、負側への1イベントを比較します。
既定は左前脚の膝です。`--leg` で6脚のいずれか、`--joint-profile` で既存の関節設定を選べます。
関節トルクの符号、無入力での能動筋力ゼロ、受動対照との差、ソルバ警告、100 µsの時計一致を記録します。
筋肉の数値や身体パラメータを、歩けるように自動調整する処理はありません。

出力は `result.json` と `observation.npz` です。身体・コード・観測のハッシュ、依存パッケージの版、
使用した仮定、関節パラメータ、メッシュのハッシュ、観測配列中の関節位置、対照条件が保存されます。失敗時には成功扱いせず、失敗理由を `result.json` に残して非ゼロ終了します。
既存の出力ディレクトリは上書きしません。各対照は最大0.1秒、壁時計予算は最大600秒です。
壁時計予算はステップ間の協調的確認であり、インポートや素材取得を中断するOSの強制タイムアウトではありません。
CIには別途15分のジョブ上限があります。CIでは6脚を順に検査し、1脚が失敗しても残りの結果を収集した上で、ジョブ全体を失敗として終了します。出力は脚ごとのディレクトリに保存します。

## 検証していないこと

単一関節の応答から、同じ筋腱で体重を支持できること、遊脚・推進、正しい位相、自律歩行を推論しません。
この身体診断は現在の `antagonist` 筋モデル用です。90腱の `tendon_candidate` を校正したことにはなりません。
F01〜F16、受入用シード、`ACCEPTANCE.json` の達成率は更新しません。

次の順序は、90腱の実形状での仮想仕事検査、同一モデルでの支持・遊脚・推進の陽性対照、
公開CPGと遮断対照、神経募集と筋力位相の照合、最後に全CNS閉ループへの復帰です。

## 一次資料

- [MuJoCo: mj_applyFT / mj_integratePos](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html)
- [MuJoCo: spatial tendon経路と巻き付き](https://mujoco.readthedocs.io/en/stable/XMLreference.html)
- [固定したFlyGymの依存要件](https://github.com/NeLy-EPFL/flygym/blob/38c8ec61034cd59bc5ba0de20688d4a3c0000d60/pyproject.toml)
- [NeuroMechFlyの筋肉シミュレーション](https://neuromechfly.org/tutorials/6_muscle_imitation/)
