# 接触・支持診断の再検証

## 接触判定の修正

以前の90腱診断は `dist > 0` の接触を捨て、身体の接触記録と矛盾する場合がありました。
`organism_core.contacts.is_active_contact` に統一し、`efc_address >= 0` かつ `exclude == 0` を使用します。
これはソルバへの拘束の登録であり、接触力が閾値以上という意味ではありません。
正の表面距離でも有効なmargin接触は含め、gap中の検出のみの接触は除きます。
`mj_collision` の直後ではなく、拘束生成後の `mj_fwdPosition` / `mj_forward` 等の状態で使います。
Bodyの足接触・荷重、支持行列、記録姿勢からの非足部接触診断を揃えています。
記録姿勢の再計算は接触拘束を分類するだけで、過去の接触力を復元したことにはなりません。

## 支持の状態は4種類以上に分離

- `not_eligible`: 有効な足接触がない、または非足部の接触がある。立位不能の証拠ではありません。
- `feasible_at_tested_pose`: 試した姿勢・制約下で静的釣り合いが可能。動的に維持できるとは限りません。
- `infeasible_under_declared_constraints`: 線形計画がその姿勢・制約では実行不可能と判定。
- `inconclusive` 等: 時間切れ、数値問題、返された解の検査失敗。実行不可能と区別します。

力上限、引張のみの筋力、圧縮のみの地面反力、保守的な摩擦ダイヤモンドを維持します。
摩擦なしの接触に接線力を許可しません。この診断は水平地面用で、傾斜面は明示的に拒否します。
関節摩擦・関節制限の反力、接触のねじり・転がりモーメントは変数に含みません。
不成立の場合には最小釣り合い誤差と該当関節を記録しますが、これを成功率に変換しません。
観測NPZには支持の行列、目標一般化力、上限、摩擦、姿勢を保存し、SciPyだけでも再計算できます。

## 高速経路

`HybridMuscles.command` を既に監査したCSR経路へ接続します。旧Cartesian計算は独立な検算として残します。
能動自由度に反映できない力を黙って捨てず、許容値を超えたら失敗させます。
神経・筋肉のパラメータ、外部素材、固定配線、受入条件は変更しません。
コード識別子が変わるので旧チェックポイントを強制復元せず、元の版で保持してください。

## 実行

Python 3.12のリポジトリ環境で、新規の作業先を指定します。

```bash
python -m pip install -e ".[dev,neural,body,mechanics]"
python -m pytest -q
python scripts/run_mechanical_study.py --workspace work/contact-retest --duration 0.25
```

CIは4環境のテスト、6脚のNMJ対照、90腱の4姿勢監査と短時間の人工張力対照を実行します。
この文書は手順と判定範囲を定義するもので、特定のCI実行の成功を事前に主張しません。
全CNS閉ループ・自律歩行・生物学的再現の認定ではありません。

## 一次資料

[MuJoCo contact](https://mujoco.readthedocs.io/en/stable/computation/index.html#contact) と
[mjContact / mjModel](https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html) を参照。
