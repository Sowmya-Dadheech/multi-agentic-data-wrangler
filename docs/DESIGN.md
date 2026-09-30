# Design notes

Why the system looks the way it does, and what was considered instead.

## 1. Orchestration: LangGraph

| Option | Strength | Why not here |
|---|---|---|
| **LangGraph** ✅ | Explicit state machine, cycles, conditional edges, checkpointer, `interrupt()`, long-term `Store` | More boilerplate than role-based frameworks |
| LangChain chains (LCEL) | Simple composition | DAG only; retry loops and pauses become ad-hoc code |
| CrewAI | Very fast to prototype role-based crews | Less direct control of routing; "code never runs without passing the guard" should be structural |
| AutoGen / AG2 | Strong conversational multi-agent patterns, built-in executors | Control flow emerges from conversation; the pipeline order should be enforced, not negotiated |
| Airflow / Prefect / Temporal | Scheduling, retries, observability | Not LLM-aware (no agent state, HITL, memory). These complement the graph: run `wrangle run` inside a scheduled task |

**Routing is plain Python.** Functions such as `route_after_validate` read the state and pick the next node. No LLM supervisor decides whether the guard runs.

## 2. State design

- Only a `source_id` goes into state. Connection strings (which may carry credentials) live in the per-invocation `Context`, which is never checkpointed.
- Data never goes into state, only file paths. State is checkpointed after every node, so a DataFrame there would bloat storage and leak data into checkpoints.
- Append-only channels (`errors`, `events`, `usage`) use reducers, so nodes add entries instead of overwriting each other.

## 3. Short-term vs long-term memory

| | Checkpointer (`SqliteSaver`) | Store (`SqliteStore`) |
|---|---|---|
| Scope | One run (`thread_id`) | All runs |
| Holds | Every intermediate state | Approved plans + code, baselines, versions |
| Enables | Resume after crash, pause for review, time travel (`Wrangler.history`) | Skipping the planner on repeat batches, hints, drift detection |

Memory hygiene:

- **Poisoning:** only validated or human-approved fixes are written.
- **Staleness:** a cached fix that fails validation is demoted.
- **Versioning:** old versions are kept in `history`.
- **Scope:** namespaces are per source.

## 4. Fingerprint

`sha256(sorted(normalized_name + ":" + dominant_type_family))[:16]`

- Normalized names make it robust to cosmetic renames of the same schema.
- The dominant **family** (`number`/`date`/`bool`/`str`), not the exact type distribution, is used because distributions wobble between samples. Hashing them would make every batch a miss.
- A `1/0 + yes/no` column resolves to `bool` in the profiler, so the fingerprint doesn't flip between `bool` and `number` from batch to batch.
- Consequence: structural drift causes a memory miss, while value drift (null jump, new categories) doesn't. That's why `detect_drift` runs on every exact hit and downgrades it to "re-plan with a hint".

## 5. Guardrails

Allowlists where possible (plan actions, Mongo stages, names in generated code), blocklists as a second layer (banned attributes and calls). The AST guard rejects:

- imports, `global`, `while`, `try`, `lambda`, `class`
- attributes starting with `_` (this closes the `().__class__.__bases__[0].__subclasses__()` family)
- `getattr` / `eval` / `exec` / `open`
- file I/O methods
- `df.eval` / `df.query` (they hide code inside strings the AST can't see)
- any name not in the allowlist and not bound locally

SQL is parsed with `sqlglot`, never regex, because regex is fooled by comments, casing and string literals. Only `SELECT`, `CREATE TABLE clean_* AS SELECT` and `DROP TABLE clean_*` are allowed. Referenced tables must be the job's source or outputs.

The sandbox assumes the guard failed:

- `python -I` (isolated mode), empty environment, throwaway working directory
- RLIMIT CPU / address space / file size, wall-clock timeout, own process group
- restricted `__builtins__` (no `__import__`), and a socket kill switch installed before the untrusted code runs

`DockerExecutor` adds `--network none --read-only --cap-drop ALL --pids-limit --user 1000`.

Humans can approve imperfect **results** (a validation failure) but not code that failed the guard or crashed. `human_review_node` enforces this.

## 6. Validation

| Check | Catches |
|---|---|
| Row count unchanged | Accidental filtering / joins |
| New nulls ≤ 2% (excluding existing nulls and null tokens), with lost values as examples | Silent coercion, the most common wrangling bug |
| Target dtype reached | Plans that "ran" but did nothing |
| Untouched columns byte-identical | Collateral damage |
| Dates within 1900–2100 (warning) | Wrong century / epoch parsing |

Validation runs twice: on the sample in the sandbox, then on the full data after apply. The full-data run is what caught the Postgres `TRIM` vs tab bug.

## 7. Cost

| Lever | Effect |
|---|---|
| Profiles, not rows, go to the model | The prompt is proportional to the number of problem columns, not the number of rows |
| Only problem columns are sent | Clean columns cost nothing |
| Rule-based draft included in the prompt | The model confirms or adjusts instead of starting from scratch |
| Templates (`wr.*`) | Little free-form code to generate, so little to repair |
| Memory | Repeat batches make zero planner calls |
| Bounded retries + `recursion_limit` | Worst-case spend is capped |

## 8. Things deliberately not done

- **No MCP for data access.** Ingestion is deterministic and the code, not the model, decides which queries run. MCP would make sense to expose the pipeline as a tool to other agents.
- **No LLM supervisor.** See §1.
- **No row dropping, ever.** Deduplication and filtering are decisions for a human or a data contract, not a cleaner.
