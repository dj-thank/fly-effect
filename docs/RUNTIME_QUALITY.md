# Runtime integrity and dispatch performance

## Scope

Software-only change based on main `89d59fc56c3e4f3b0f9a26725ce4ee55fdf1ee36`.
It does not change neural IDs/edges/weights, model parameters, force limits,
`ACCEPTANCE.json`, graph lock, source assets, or the body research protocol.
Code identity naturally changes; do not relabel an incompatible checkpoint.

## Checkpoints

Valid schema-1 archives remain readable. Dictionary-key duplication, duplicate
ZIP/JSON members, unsupported nodes, malformed scalars, reused/missing array
references, orphan arrays and nonfinite JSON constants now fail explicitly.
A save failure during serialization, flushing or replacement leaves the earlier
checkpoint intact and removes the temporary file. Hashing happens on the completed
temporary file before atomic replacement. Loading verifies and reads one open file,
not two independently opened pathnames. `fsync` flushes file contents; directory
crash durability and hostile-archive resource isolation are not claimed.

## Motor dispatch

Build a required-for-walking index once after motor mapping. Decoding no longer
scans all annotation rows for each unmapped motor event. Mapping choices and all
counters are unchanged, including repeated events and outside-scope events.
Pandas is imported only while constructing the data-backed map; pure dispatch
checks require NumPy, not a biological dataset or pandas.

A synthetic microbenchmark on Python 3.13.5 / Linux used 100 annotation rows,
140 unmapped events per tick, 1,000 ticks per repeat and seven repeats. The median
was 0.250364 s for the previous scan and 0.020487 s for indexed dispatch (12.22x).
This isolates a worst-case dispatch workload; it is NOT a measured full-CNS or
whole-simulator speedup. The deterministic regression compares 100 mixed random
batches, all outputs and every counter. No wall-clock threshold is used in CI.

Reproduce the focused benchmark from the repository root:

```python
import runpy, statistics, timeit
m = runpy.run_path('tests/unit/test_motor_dispatch.py')
fixture, reference = m['fixture'], m['reference']
a, b = fixture(), fixture()
fired = list(range(30, 100)) * 2
old = timeit.repeat(lambda: reference(a, fired), number=1000, repeat=7)
new = timeit.repeat(lambda: b.decode(fired), number=1000, repeat=7)
print(statistics.median(old), statistics.median(new))
```

## CI

Retain all four existing required matrix job names and their test/build checks.
Run branch changes via `pull_request` and main via `push`, avoiding duplicate
push+PR matrices on open branches. Draft PRs are checked; unpublished branches
without a PR no longer trigger this workflow. Cache pip downloads keyed by
`pyproject.toml`, cancel superseded runs, cap each job at 15 minutes, and retain
JUnit plus one distribution archive. No paid service or recurring schedule is added.

## Local verification before submission

41 added tests passed. The broader PR #17 source snapshot with these changes gave
224 passed and eight backend-dependent modules skipped; no local MuJoCo, FlyGym
or Brian2 execution is claimed. Those diagnostic-only files are NOT included in
this main-based software PR. Its own four-environment CI results must be checked
on the exact submitted commit. The POSIX pathname-replacement test is explicitly
skipped on Windows; other integrity tests run on both operating systems.
