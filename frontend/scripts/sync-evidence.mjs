// Copy only a named allowlist of versioned aggregates; never open artifacts or raw data.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
import Papa from 'papaparse';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const sources = [];
function read(name, csv = false) {
  const source = `docs/results/${name}`;
  const text = fs.readFileSync(path.join(root, source), 'utf8');
  sources.push({ path: source, sha256: crypto.createHash('sha256').update(text).digest('hex') });
  if (!csv) return JSON.parse(text);
  const parsed = Papa.parse(text, { header: true, skipEmptyLines: true });
  if (parsed.errors.length) throw new Error(`Invalid aggregate CSV: ${name}`);
  return parsed.data;
}
const release = read('release-04.json');
const freeze = read('release-04-freeze.json');
const n = (value) => {
  if (value === '') return null;
  if (typeof value !== 'string' || !Number.isFinite(Number(value)))
    throw new Error('Invalid numerical aggregate.');
  return Number(value);
};
const experiment = (r) => ({
  name: r.model,
  rows: n(r.scored_rows),
  cost: n(r.cost),
  cost_per_row: n(r.cost_per_row),
  recall: n(r.mean_recall),
  precision: n(r.mean_precision),
  average_precision: n(r.mean_average_precision),
});
const reference = {
  release_id: release.release_id,
  freeze_sha256: release.official_test.freeze_sha256,
  threshold: release.decision.threshold,
  calibration: release.decision.calibration,
  official_test: release.official_test.metrics,
  validation: release.validation,
  calibration_cv: release.calibration_cv,
  missingness: read('release-04-missingness.csv', true).map((r) => ({
    lower: n(r.missing_lower),
    upper: n(r.missing_upper),
    rows: n(r.rows),
    positives: n(r.positive),
    fp: n(r.fp),
    fn: n(r.fn),
    recall: n(r.positive) === 0 ? null : n(r.recall),
  })),
  permutation: read('release-04-permutation-groups.csv', true)
    .map((r) => ({
      name: r.prefix,
      columns: n(r.columns),
      cost_delta: n(r.cost_increase_per_row_mean),
      std: n(r.cost_increase_per_row_std),
    }))
    .sort((a, b) => b.cost_delta - a.cost_delta)
    .slice(0, 5),
  comparison: read('comparison-02.csv', true).map(experiment),
  optimization: read('optimization-03.csv', true).map(experiment),
  sources,
};
if (
  freeze.release_id !== reference.release_id ||
  freeze.threshold !== reference.threshold ||
  reference.official_test.cost !==
    10 * reference.official_test.fp + 500 * reference.official_test.fn ||
  reference.missingness.reduce((s, r) => s + r.rows, 0) !== reference.validation.release.rows
)
  throw new Error('Reference aggregates are inconsistent.');
const target = path.join(root, 'frontend/public/evidence');
fs.mkdirSync(target, { recursive: true });
fs.writeFileSync(path.join(target, 'reference.json'), JSON.stringify(reference, null, 2) + '\n');
console.log('Synced 6 aggregate sources; no raw data/model or individual predictions copied.');
