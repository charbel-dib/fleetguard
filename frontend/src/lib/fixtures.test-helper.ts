import type { ModelInfo, SensorRow } from './contracts';
export const model: ModelInfo = {
  service_version: '0.6.0',
  contract_version: 1,
  model: {
    release_id: 'fixture-release',
    name: 'xgboost_release',
    dataset_kind: 'synthetic_smoke',
    pipeline_sha256: 'a'.repeat(64),
    freeze_sha256: 'b'.repeat(64),
    threshold: 0.4,
    calibration: 'raw',
    policy: 'challenge_cost',
    score_semantics: 'fixture',
  },
  feature_names: ['a', 'b'],
  limits: { max_batch_rows: 2, max_request_bytes: 200, feature_abs_max: 3.4028234663852886e38 },
  costs: { false_positive_cost: 10, false_negative_cost: 500 },
};
export const rows: SensorRow[] = [
  { a: 0.1, b: null },
  { a: 0.5, b: 3 },
  { a: 1, b: null },
];
export function result(input: SensorRow[]) {
  return {
    model: model.model,
    rows: input.length,
    predictions: input.map((r, i) => ({
      row_index: i,
      positive_score: r.a ?? 0,
      predicted_label: (r.a ?? 0) >= model.model.threshold ? 'pos' : 'neg',
      missing_fraction: r.b === null ? 0.5 : 0,
    })),
  };
}
