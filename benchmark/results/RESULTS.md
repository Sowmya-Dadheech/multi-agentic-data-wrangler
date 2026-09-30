# Benchmark results (planner: `heuristic`)

36 synthetic datasets across 6 domains and 3 source types (CSV, SQLite, MongoDB), 233,000 rows in the first pass, with injected corruption and full ground truth. Reproduce with `python benchmark/run_benchmark.py`.

| metric | first contact | repeat batch | drift |
|---|---:|---:|---:|
| runs | 36 | 36 | 12 |
| succeeded without a human | 86.1% | 100.0% | 100.0% |
| succeeded on the first attempt, no human | 47.2% | 100.0% | 66.7% |
| paused for human review | 13.9% | 0.0% | 0.0% |
| succeeded after review (simulated approver) | 100.0% | 100.0% | 100.0% |
| mean attempts | 1.47 | 1.0 | 1.42 |
| exact memory hits | 0.0% | 83.3% | 16.7% |
| similar-schema hints | 83.3% | 16.7% | 83.3% |
| planner calls | 45 | 6 | 15 |
| issue detection precision | 98.5% | 98.5% | 100.0% |
| issue detection recall | 98.8% | 98.8% | 99.5% |
| dirty cells repaired correctly | 99.1% | 99.1% | 99.5% |
| clean cells damaged (collateral) | 0.0% | 0.0% | 0.0% |
| unrecoverable junk nulled (not guessed) | 100.0% | 100.0% | 100.0% |
| column renames mapped correctly | 94.7% | 94.7% | 96.3% |
| cached fixes demoted | 0 | 0 | 1 |
| median seconds per run | 1.25 | 0.96 | 1.06 |

**Memory effect (pass 1 → pass 2):** planner calls 45 → 6 (86.7% fewer); prompt tokens 97,922 → 10,917 (~9.0× fewer).

**Guardrails:** 46/46 red-team programs blocked (100.0%); every one of the 84 generated transforms passed through the AST/query guard and the sandbox before touching data.

**Why runs paused for review:** planner confidence 0.40 below 0.6 (3), planner confidence 0.45 below 0.6 (2)

**Repair accuracy by column type (first pass):** bool 100.0%, category 100.0%, date 100.0%, datetime 100.0%, float 91.4%, id 100.0%, int 100.0%, str 100.0%

**Drift pass:**

| dataset | drift | memory | status | attempts | cached fix demoted |
|---|---|---|---|---:|---|
| customers_0 | new_column | similar | succeeded | 1 |  |
| customers_3 | format_change | similar | succeeded | 1 |  |
| orders_0 | broken_field | similar | succeeded | 1 |  |
| orders_3 | new_column | similar | succeeded | 1 |  |
| patients_0 | format_change | similar | succeeded | 2 |  |
| patients_3 | broken_field | similar | succeeded | 1 |  |
| sensors_0 | new_column | similar | succeeded | 2 |  |
| sensors_3 | format_change | exact | succeeded | 1 |  |
| employees_0 | broken_field | similar | succeeded | 1 |  |
| employees_3 | new_column | similar | succeeded | 2 |  |
| products_0 | format_change | exact | succeeded | 3 | yes |
| products_3 | broken_field | similar | succeeded | 1 |  |

_Offline mode: token figures are the size of the prompt the LLM planner would have been sent (chars/4); no model was called._
