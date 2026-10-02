# 導入

基本開発・人工回路はPython 3.11または3.12、身体・全CNSはPython 3.12とGitを用意し、仮想環境でREADMEの手順を実行してください。PowerShellでActivate.ps1を実行できない場合、設定を変えず仮想環境のPythonを直接指定できます。

- 基本開発: python -m pip install -e ".[dev]"
- 人工回路: python -m pip install -e ".[dev,neural]"
- 人工力学検査: python -m pip install -e ".[dev,neural,mechanics]"
- 神経データ不要の身体診断（Python 3.12）: python -m pip install -e ".[body]"
- 身体・全CNS（Python 3.12）: python -m pip install -e ".[dev,simulation]"

固定したFlyGymのPython要件は >=3.12,<3.15 です。まず [筋力・身体の切り分け検証](MECHANICAL_VALIDATION.md) で、人工モデルの力学検査と同一六脚モデルへの対照刺激を実行できます。

simulationにはGitHub依存と、別途取得する大型アセットが必要になる場合があります。doctorは不足物を表示し、--check-dataで外部グラフのハッシュも検査します。診断は実験の成功を認定しません。デモには新しい出力ディレクトリを指定してください。

## 全系の実験

完全自動の生物データ取得・コンパイル手順はまだありません。[DATA.md](DATA.md)の互換スナップショットと補助データを準備する必要があります。個人のCompletion Systemには依存しない構成へ移植していますが、全系導入は研究開発者向けです。

FLY_EFFECT_HOMEに作業先、FLY_EFFECT_DATA_DIRにデータの親ディレクトリを指定します。既定値は現在のディレクトリと、そのdataフォルダです。大きな実験前にデータ・空き容量・メモリ・時間上限を確認し、1件ずつ起動してください。

入口は python organism.py --help。準備後にrun/resumeを使います。view/recordの統合UIは未完成です。記録済み姿勢はrender_pose.py、動画はrender_recorded_motion.pyで描画できます。動画にはFFmpegが必要です。

audit_source_tendons.py → transfer_six_leg_tendons.pyは、FlyGymの元筋モデルから候補を作る研究用手順です。infer_motor_targets.pyは対応仮説を生成します。生成物はHOME/runsに置き、別環境では作り直します。

問題があればバージョンと最小再現手順をIssueに報告してください。既知の再現環境の固定値はpyproject.tomlにあります。


全系runの壁時計上限はチャンク間で確認する協調的な制限です。メモリ上限をOSレベルで強制する機能は未実装です。初回から長い全CNS実験を回さず、人工回路と小さな再現条件から始めてください。
