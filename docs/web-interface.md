# Interface locale — FleetGuard 0.6

The React/TypeScript application has three views: alert review for unlabeled uploaded measurements,
diagnostics of the published reference release, and the historical model comparison/optimization.
It uses the existing frozen HTTP contract; no model search or official-test evaluation is added.

## Running

Install Node.js 24 LTS (supported range: >=22.12, <25) and the Python serving environment. From
`frontend`, run `npm ci`, `npm run check`, `npm test`, then `npm run build`.
The production build can be served by the existing CLI:

```bash
uv run --no-sync python -m fleetguard serve --release artifacts/releases/YOUR_RELEASE --web-dir frontend/dist
```

Open <http://127.0.0.1:8000>. `/health`, `/v1`, `/docs` and `/openapi.json` retain their API behavior.
The static root mount is optional, registered last, and restricted to the chosen build directory.
The directory must contain `index.html` and `assets/`; it cannot point to an incomplete build.
`FLEETGUARD_WEB_DIR` is the equivalent environment setting.

For development, start the API on port 8000, then `npm run dev` from `frontend` and open
<http://127.0.0.1:5173>. Vite proxies `/health` and `/v1` to localhost:8000; the browser uses relative
paths. `FLEETGUARD_API_TARGET` can change that development-only proxy target before starting Vite.
Both flows use a same-origin browser contract, without adding broad CORS permissions to the API.
`npm run preview` previews static assets only; use the integrated Python server for production
HTTP/inference verification. The Docker API remains serving-only in this increment; public frontend
hosting and routing belong to the deployment stage.

## Input and inference behavior

The browser obtains readiness, feature names, frozen identity and limits from the API. It never
hardcodes the official reference threshold or assumes that every local release uses sigmoid.
Official APS releases require 170 sensors; an explicit synthetic server is visibly labeled.

CSV parsing uses Papa Parse with comma delimiters and normal quoting. Exactly the active feature
names are required, in any order. Duplicate/missing/unknown headers, `class`/IDs, wrong field counts,
numeric strings with units, booleans, hex, NaN/Infinity and values outside float32 range are refused
before inference. Empty cells, `na` and `null` become JSON null. Empty sensor rows remain present;
truly blank CSV lines are skipped. Row positions refer to parsed measurements, not physical CSV
line numbers or truck IDs.

The local browser caps input at 5 MiB and 5,000 measurement rows. It splits sequential requests by
both the API row cap and **actual UTF-8 JSON body bytes**. Each batch response is validated for count,
order, score bounds, threshold-consistent labels, missingness and unchanged model identity. Request
indices are restored to global CSV positions. The model/schema is rechecked before sending a lot.

The UI shows progress, supports cancellation and applies a 30-second timeout per HTTP request.
Only a complete successful lot is displayed. Error, timeout, cancellation or model replacement
discards partial outcomes; no automatic prediction retry is performed. Cancellation of browser
fetch does not guarantee that already admitted server computation stops. Inputs remain available
for a deliberate new attempt. Reconnecting clears the old schema, input, results and annotations.

An example/download is generated from the active feature names, using explicit null/zero values.
It is an artificial contract example, not real truck data or measured diagnostic performance.
No Scania sensor records or fitted models are included in the frontend.

## Review and interpretation

The lot summary reports measurement count, number/fraction predicted pos and average missingness.
It does not compute recall, precision or false-negative cost from unlabeled input. Pos follows the
frozen `score >= threshold` rule; neg describes failure outside APS, not a healthy truck.

Users can filter/sort/page decisions, inspect input sensors and hashes in a native dialog, and mark
rows examined. Review marks survive navigation between views but reset for new input/analysis.
They are held in React memory only. The optional JSON export preserves model identity and results,
zero-based `row_index` and reviewed flags; raw input sensors are not exported. The UI presents
one-based measurement numbers for reading. Closing/reloading the page clears session state.

The CSV is read in the browser. Only numerical/null sensor rows are sent to the chosen local API;
the application does not upload or persist the original file. There is no telemetry, remote font
fetch, localStorage or database in this UI. This does not define a general public-service retention
policy; hosting and access controls are future deployment work.

## Published evidence

`frontend/scripts/sync-evidence.mjs` reads an explicit allowlist of six existing files under
`docs/results`: the release/freeze JSON and comparison, optimization, missingness and grouped
permutation CSVs. It copies only aggregate fields to an ignored generated reference JSON, recording
source hashes. The same step runs before development/build. It never reads raw data, local model
artifacts, individual predictions or the official test CSV.

The reference views always identify their release and explicitly separate it from the active API
model and imported lot. Final-test recall/precision have denominators and a confusion matrix;
challenge cost is not currency. Calibration comparison uses retained validation. Missingness slices
with zero positives show recall as undefined. Permutation shows sensitivity, not causal explanations.
The experiment table compares candidates **within** its selected CV stage; it labels fold averages
versus aggregate cost and explains selection optimism and differing protocols between stages.

## Validation

28 Vitest cases cover strict CSV parsing, row/byte caps, examples, batching/order, cancellation,
HTTP failures, contract violations and model identity changes. Seven Playwright workflows exercise
the **built** UI against a real synthetic HTTP server, including multi-request inference, CSV import,
export and review persistence, drift/error rejection, cancellation, offline/reconnect, keyboard
dialog focus/Escape, axe scans and mobile overflow checks. Synthetic tests are software evidence.

Python adds static-root/API isolation, traversal/non-exposure and incomplete-build/env checks.
The complete Python suite passes 104 tests. Existing offline smokes are preserved. The CI frontend
job installs Node 24, core + serving Python dependencies, builds and runs browser tests, without
retraining a Scania model. Browser results were checked locally under Linux/Chromium 153; CI uses
Playwright's own Chromium and Windows reproduction remains to be confirmed on the user's machine.
A clean automated accessibility scan is scoped to tested screens; it is not a full WCAG certification.

[Local validation record](results/web-06.json) and [desktop](results/web-06-home.png) /
[mobile](results/web-06-mobile.png) captures document the installed-wheel reference check. The mobile
lot uses artificial null/zero inputs. Captures do not establish the identity or results of another
user's release; the live model card supplies that identity.

See [UPDATE_06.md](../UPDATE_06.md) for Windows environment/execution-policy setup and Git flow,
and [ADR 0006](decisions/0006-separate-unlabeled-review-from-reference-evidence.md).
