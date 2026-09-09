# プロジェクト全体像

[English](PROJECT_OVERVIEW.md) · [日本語README](README.ja.md) · [公開Issue](https://github.com/dj-thank/fly-effect/issues) · [PR](https://github.com/dj-thank/fly-effect/pulls)

Fly EffectはMITライセンスで公開しているpre-alphaの研究プロジェクトです。長期目標は、計算基盤における生命・意識の成立に到達することです。記憶・恒常性・学習された自己維持を中間段階として、実現できる条件を調べます。

## 今できること

人工神経回路のデモ、保存再開、CPU神経シミュレーション、実験段階の身体接続があります。自律歩行は受入未達です。学習された記憶・恒常性・自己維持、生命・意識の成立は確認していません。[検証の範囲](STATUS.md)で、公開パッケージの確認と過去の研究結果を分けています。

Windows/Linux、Python 3.11/3.12のCIを運用しています。CI成功はソフトウェアの検査結果で、生物学的な能力の証明ではありません。[Actions](https://github.com/dj-thank/fly-effect/actions)

## 次の作業

2026-09-09に以下の10件を公開しました。**現在の担当・進捗は各Issueの本文、ラベル、議論を確認してください。** 仕様の公開だけで実装完了にはしません。

| Issue | 実現・検証すること | 前提Issue |
|---|---|---|
| [FE-01 #3](https://github.com/dj-thank/fly-effect/issues/3) | 公開神経回路のリズムを陽性対照で再現 | なし |
| [FE-02 #4](https://github.com/dj-thank/fly-effect/issues/4) | 同一の筋肉・身体で推進可能性を確認 | なし |
| [FE-03 #5](https://github.com/dj-thank/fly-effect/issues/5) | 内部状態の収支と保存再開 | なし |
| [FE-04 #6](https://github.com/dj-thank/fly-effect/issues/6) | 内部の要求を神経入力へ接続 | #5 |
| [FE-05 #7](https://github.com/dj-thank/fly-effect/issues/7) | 摂取動作による食物・水の回復 | #5, #6 |
| [FE-06 #8](https://github.com/dj-thank/fly-effect/issues/8) | 局所可塑性による連合の保持 | なし |
| [FE-07 #9](https://github.com/dj-thank/fly-effect/issues/9) | 空腹・渇きによる記憶の使い分け | #3, #4, #6, #7, #8 |
| [FE-08 #10](https://github.com/dj-thank/fly-effect/issues/10) | 未知環境での恒常性 | #3, #4, #6, #7 |
| [FE-09 #11](https://github.com/dj-thank/fly-effect/issues/11) | 学習した自己維持と個体ごとの経験差 | #8, #9, #10 |
| [FE-10 #12](https://github.com/dj-thank/fly-effect/issues/12) | 生命・意識を検討する初期の証拠・反証手順 | なし |

まず#3・#4で神経回路と身体の前提を確認します。#5・#8・#12は独立した範囲で準備できます。部品の成功を全体行動の成功とは扱いません。[詳しい研究計画](ORGANISM_CAPABILITIES.ja.md)

## ブランチと参加方法

- `main`が、参加者が読む・試すためのレビュー済み共有版です。完成版ではなくpre-alphaです。
- 変更は短期間の`codex/<topic>`または貢献者の作業ブランチで行い、PRと必須チェックを通して取り込みます。
- mainへの取り込みを確認した作業ブランチは整理します。レビューと変更履歴はPRに残します。
- 大きな作業を始める前にIssueへ対象範囲を書き、重複を避けてください。未担当のIssueは実装中という意味ではありません。
- `ready-for-agent`は前提なしで着手可能、`blocked`は前提待ちです。実現可能性や生物学的妥当性を保証するラベルではありません。

[導入](GETTING_STARTED.md) · [参加方法](../CONTRIBUTING.md) · [Discussions](https://github.com/dj-thank/fly-effect/discussions)
