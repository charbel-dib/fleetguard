# ADR 0006 — separate unlabeled review from reference evidence

Status: accepted for increment 06.

The frozen service may load a user's local release with a different threshold/calibration from the
published reference. Uploaded measurements have no labels. A UI that places historical recall next
to the active lot without provenance would imply a performance claim that has not been measured.

Use three explicit views. Alert review binds to live model identity/schema/limits and reports only
observable lot quantities. Diagnostics and experiment views read a fixed allowlist of versioned
aggregate evidence, name their reference release and state that the numbers do not describe the
active model or upload. They remain readable when the API is unavailable.

Validate CSV locally, split requests by row/byte caps, and validate every response. Abort the whole
display outcome on partial failure or identity/schema replacement. Preserve row order/index and
export model identity with deliberate local review annotations. Do not tune thresholds from the UI,
send labels, persist uploaded data or report fabricated precision/recall.

Serve the built static frontend optionally from the same local Python origin, preserving the API
routes. Use Vite's development proxy for the same relative route contract. This avoids introducing
public CORS/access policy before a deployment exists. Cloud image composition, authentication,
promotion and rollback remain the next increment's decisions.

Consequences: local review is useful and auditable without confusing synthetic examples with Scania
metrics. A native dialog supports keyboard focus behavior; responsive tables retain internal scroll.
Production-built UI + actual synthetic API tests validate integration rather than relying entirely
on mocked fetch. The reference JSON is reproducibly generated from committed sources, not another
editable collection of copied metric constants.
