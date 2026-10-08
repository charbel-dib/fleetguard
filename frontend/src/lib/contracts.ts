export type SensorRow = Record<string, number | null>;
export type Identity = {
  release_id: string;
  name: string;
  dataset_kind: 'official_aps' | 'synthetic_smoke';
  pipeline_sha256: string;
  freeze_sha256: string;
  threshold: number;
  calibration: 'raw' | 'sigmoid';
  policy: 'challenge_cost';
  score_semantics: string;
};
export type ModelInfo = {
  service_version: string;
  contract_version: 1;
  model: Identity;
  feature_names: string[];
  limits: { max_batch_rows: number; max_request_bytes: number; feature_abs_max: number };
  costs: { false_positive_cost: number; false_negative_cost: number };
};
export type Prediction = {
  row_index: number;
  positive_score: number;
  predicted_label: 'pos' | 'neg';
  missing_fraction: number;
};
export type Result = { model: Identity; rows: number; predictions: Prediction[] };

export function identityKey(m: Identity): string {
  return JSON.stringify([
    m.release_id,
    m.name,
    m.dataset_kind,
    m.pipeline_sha256,
    m.freeze_sha256,
    m.threshold,
    m.calibration,
    m.policy,
    m.score_semantics,
  ]);
}
function record(x: unknown): x is Record<string, unknown> {
  return typeof x === 'object' && x !== null && !Array.isArray(x);
}
const finite = (x: unknown): x is number => typeof x === 'number' && Number.isFinite(x);
export function validateIdentity(x: unknown): Identity {
  if (
    !record(x) ||
    !['release_id', 'name', 'score_semantics'].every((k) => typeof x[k] === 'string' && x[k]) ||
    !['pipeline_sha256', 'freeze_sha256'].every(
      (k) => typeof x[k] === 'string' && /^[a-f0-9]{64}$/.test(x[k]),
    ) ||
    !finite(x.threshold) ||
    x.threshold < 0 ||
    x.threshold > 1 + Number.EPSILON ||
    !['raw', 'sigmoid'].includes(String(x.calibration)) ||
    x.policy !== 'challenge_cost' ||
    !['official_aps', 'synthetic_smoke'].includes(String(x.dataset_kind))
  ) {
    throw new Error('Identité de modèle incompatible avec le contrat v1.');
  }
  return x as Identity;
}
export function validateModel(x: unknown): ModelInfo {
  if (
    !record(x) ||
    x.contract_version !== 1 ||
    typeof x.service_version !== 'string' ||
    !Array.isArray(x.feature_names) ||
    !x.feature_names.length ||
    !x.feature_names.every((n) => typeof n === 'string' && n.length > 0) ||
    new Set(x.feature_names).size !== x.feature_names.length ||
    !record(x.limits) ||
    !Number.isInteger(x.limits.max_batch_rows) ||
    Number(x.limits.max_batch_rows) < 1 ||
    Number(x.limits.max_batch_rows) > 1024 ||
    !Number.isInteger(x.limits.max_request_bytes) ||
    Number(x.limits.max_request_bytes) < 1 ||
    Number(x.limits.max_request_bytes) > 16777216 ||
    !finite(x.limits.feature_abs_max) ||
    x.limits.feature_abs_max <= 0 ||
    !record(x.costs) ||
    !['false_positive_cost', 'false_negative_cost'].every(
      (k) =>
        finite((x.costs as Record<string, unknown>)[k]) &&
        Number((x.costs as Record<string, unknown>)[k]) >= 0,
    )
  ) {
    throw new Error('Schéma de service incompatible avec le contrat v1.');
  }
  const model = validateIdentity(x.model);
  if (model.dataset_kind === 'official_aps' && x.feature_names.length !== 170) {
    throw new Error('Une release APS officielle doit déclarer 170 capteurs.');
  }
  return x as ModelInfo;
}
export function validateResult(x: unknown, model: Identity, input: SensorRow[]): Result {
  if (
    !record(x) ||
    x.rows !== input.length ||
    !Array.isArray(x.predictions) ||
    x.predictions.length !== input.length
  ) {
    throw new Error('Nombre de résultats incompatible avec le lot envoyé.');
  }
  const received = validateIdentity(x.model);
  if (identityKey(received) !== identityKey(model)) {
    throw new Error(
      'Le modèle a changé pendant l’analyse. Le résultat entier est écarté ; reconnectez le service.',
    );
  }
  x.predictions.forEach((p, i) => {
    const missing =
      Object.values(input[i]).filter((v) => v === null).length / Object.keys(input[i]).length;
    if (
      !record(p) ||
      p.row_index !== i ||
      !finite(p.positive_score) ||
      p.positive_score < 0 ||
      p.positive_score > 1 ||
      p.predicted_label !== (p.positive_score >= model.threshold ? 'pos' : 'neg') ||
      !finite(p.missing_fraction) ||
      Math.abs(p.missing_fraction - missing) > 1e-12
    ) {
      throw new Error('Réponse de prédiction incohérente avec le contrat figé.');
    }
  });
  return x as Result;
}
