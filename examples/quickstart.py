"""Use the pipeline from Python instead of the CLI.

    python examples/quickstart.py
"""

import json
from pathlib import Path

from wrangler import Settings, Wrangler
from wrangler.datagen import CorruptionProfile, load_into, make_dataset

work = Path("demo_output")
work.mkdir(exist_ok=True)

# 1. make a messy CSV (synthetic data with known ground truth)
ds = make_dataset("customers", 3000, seed=7, cp=CorruptionProfile(date_style="mixed_eu"))
source = load_into(ds, "csv", "customers", str(work))
print("messy input:\n", ds.dirty.head(5).to_string(), "\n")

# 2. run the graph. hitl="interrupt" pauses for a human when needed
w = Wrangler(Settings(home=work / ".wrangler"), hitl="interrupt")
target = json.loads(Path(__file__).with_name("target_schema.json").read_text())
result = w.run(source, target_schema=target)

if result.interrupted:  # a reviewer would look at result.interrupt here
    print("paused for review:", result.interrupt["reason"])
    result = w.resume(result.run_id, {"action": "approve"})

for e in result.state["events"]:
    print(f"{e['node']:>9} | {e['msg']}")

# 3. inspect what was learned
print("\nstored fixes:", [(m["_source"], m["fingerprint"], m["version"]) for m in w.memory.list_all()])
print("\ncleaned output:", result.state["output_ref"])
