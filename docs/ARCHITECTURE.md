# アーキテクチャ

    環境 → 感覚アダプター → 実ニューロンID → 神経計算
      ↑                                      ↓
    身体・接触 ← 関節トルク ← 筋活動・張力 ← 運動出力

- organism_core/brain.py: 固定グラフとCPU LIF計算。Eonを動的ロードしない公開用アダプター。
- body.py: 実験的な六脚身体と接触。
- tendon_*.py / muscles.py: 未校正の筋肉・腱モデル。
- proprioception.py / load_receptors.py: 関節・荷重からの感覚仮説。
- checkpoint.py: 配列・型付きJSONによる状態保存。
- config.py: FLY_EFFECT_HOME / FLY_EFFECT_DATA_DIRのパス境界。
- fly_effect.py: 導入診断と人工回路デモ。
- organism.py: データ準備後の実験用全系入口。

コード、外部データ、実行結果を分離します。HOMEは作業先、DATA_DIRはデータ置き場です。生成物と実行記録はリポジトリへコミットしません。公開用移植と過去の受入実績は別に評価します。

