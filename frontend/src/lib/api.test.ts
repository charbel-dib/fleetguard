import { afterEach, describe, expect, it, vi } from 'vitest';
import { batches, connect, predict } from './api';
import { validateModel, validateResult } from './contracts';
import { model, result, rows } from './fixtures.test-helper';
import type { SensorRow } from './contracts';
afterEach(() => vi.unstubAllGlobals());
const ok = (data: unknown) =>
  new Response(JSON.stringify(data), { headers: { 'Content-Type': 'application/json' } });
function transport(batch: (values: SensorRow[]) => Response = (values) => ok(result(values))) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url, options) => {
      if (url === '/health/ready') return ok({ status: 'ready' });
      if (url === '/v1/model') return ok(model);
      return batch(JSON.parse(options.body).rows);
    }),
  );
}
describe('bounded inference and runtime response validation', () => {
  it('honors row and actual serialized UTF-8 byte caps', () => {
    const byteCap = new TextEncoder().encode(JSON.stringify({ rows: [rows[0]] })).length;
    const groups = batches(rows, {
      ...model,
      limits: { ...model.limits, max_request_bytes: byteCap },
    });
    expect(groups.flat()).toEqual(rows);
    groups.forEach((group) =>
      expect(new TextEncoder().encode(JSON.stringify({ rows: group })).length).toBeLessThanOrEqual(
        byteCap,
      ),
    );
    expect(batches(rows, model).map((g) => g.length)).toEqual([2, 1]);
  });
  it('refuses a single row too large for the server', () => {
    expect(() =>
      batches(rows, { ...model, limits: { ...model.limits, max_request_bytes: 12 } }),
    ).toThrow('Une mesure');
  });
  it('connects and aggregates request-local row indices into global input positions', async () => {
    transport();
    const progress: number[] = [];
    expect(await connect(new AbortController().signal)).toEqual(model);
    const output = await predict(rows, model, new AbortController().signal, (n) =>
      progress.push(n),
    );
    expect(output.predictions.map((p) => p.row_index)).toEqual([0, 1, 2]);
    expect(output.predictions.map((p) => p.predicted_label)).toEqual(['neg', 'pos', 'pos']);
    expect(progress).toEqual([2, 3]);
  });
  it('discards the whole run if a later batch changes model identity', async () => {
    let calls = 0;
    transport((values) => {
      const data = result(values);
      if (++calls === 2) data.model = { ...model.model, release_id: 'changed' };
      return ok(data);
    });
    await expect(predict(rows, model, new AbortController().signal, () => {})).rejects.toThrow(
      'modèle a changé',
    );
  });
  it('detects a changed model before sending measurements', async () => {
    transport();
    const old = { ...model, model: { ...model.model, threshold: 0.9 } };
    await expect(predict(rows, old, new AbortController().signal, () => {})).rejects.toThrow(
      'modèle ou son schéma',
    );
  });
  it('handles plain-text overload errors and retains request identity', async () => {
    transport(
      () =>
        new Response('Service Unavailable', { status: 503, headers: { 'x-request-id': 'abc123' } }),
    );
    await expect(predict(rows, model, new AbortController().signal, () => {})).rejects.toThrow(
      'HTTP 503 (requête abc123)',
    );
  });
  it('supports cancellation without retrying or returning partial results', async () => {
    const controller = new AbortController();
    controller.abort();
    transport();
    await expect(predict(rows, model, controller.signal, () => {})).rejects.toMatchObject({
      name: 'AbortError',
    });
    expect(fetch).not.toHaveBeenCalled();
  });
  it('refuses swapped row positions, altered decisions and incorrect missing fractions', () => {
    const invalid = result(rows);
    invalid.predictions[0].row_index = 1;
    expect(() => validateResult(invalid, model.model, rows)).toThrow('incohérente');
    const wrongLabel = result(rows);
    wrongLabel.predictions[0].predicted_label = 'pos';
    expect(() => validateResult(wrongLabel, model.model, rows)).toThrow('incohérente');
    const wrongMissing = result(rows);
    wrongMissing.predictions[0].missing_fraction = 0;
    expect(() => validateResult(wrongMissing, model.model, rows)).toThrow('incohérente');
  });
  it('requires a supported schema, full official feature count and finite scores', () => {
    expect(() => validateModel({ ...model, contract_version: 2 })).toThrow('Schéma');
    expect(() =>
      validateModel({ ...model, model: { ...model.model, dataset_kind: 'official_aps' } }),
    ).toThrow('170');
    const invalid = result(rows);
    invalid.predictions[0].positive_score = NaN;
    expect(() => validateResult(invalid, model.model, rows)).toThrow('incohérente');
  });
});
