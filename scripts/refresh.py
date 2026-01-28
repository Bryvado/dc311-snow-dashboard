from pathlib import Path
import json
from datetime import datetime, timezone

out = Path("docs/data")
out.mkdir(parents=True, exist_ok=True)

(out / "last_refresh.json").write_text(json.dumps({
    "ok": True,
    "last_refresh_utc": datetime.now(timezone.utc).isoformat()
}, indent=2))

print("Wrote docs/data/last_refresh.json")
