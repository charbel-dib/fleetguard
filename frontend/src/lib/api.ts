import { identityKey, validateModel, validateResult } from './contracts';
import type { ModelInfo, Result, SensorRow } from './contracts';

async function request(path: string, signal: AbortSignal, payload?: unknown): Promise<unknown> {
  const controller = new AbortController();
  const cancel = () => controller.abort();
  if (signal.aborted) throw new DOMException('Aborted', 'AbortError');
  signal.addEventListener('abort', cancel, { once: true });
  let timedOut = false;
  const timeout = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, 30000);
  try {
    const response = await fetch(path, {
      signal: controller.signal,
      cache: 'no-store',
      ...(payload === undefined
        ? {}
        : {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
          }),
    });
    if (!response.ok) {
      // Uvicorn/proxies may return plain text, not the application envelope.
      const id = response.headers.get('x-request-id');
      throw new Error(
        `Le service a répondu HTTP ${response.status}${id ? ` (requête ${id})` : ''}. Aucun résultat complet n’est disponible.`,
      );
    }
    try {
      return await response.json();
    } catch {
      throw new Error('Le service a renvoyé une réponse JSON invalide.');
    }
  } catch (e) {
    if (timedOut)
      throw new Error(
        'Le service a dépassé le délai de 30 secondes. Relancez après vérification de sa disponibilité.',
      );
    if (e instanceof TypeError)
      throw new Error('Service inaccessible. Vérifiez le démarrage de l’API puis reconnectez.');
    throw e;
  } finally {
    clearTimeout(timeout);
    signal.removeEventListener('abort', cancel);
  }
}
export async function connect(signal: AbortSignal): Promise<ModelInfo> {
  const ready = (await request('/health/ready', signal)) as { status?: string };
  if (ready.status !== 'ready') throw new Error('Le modèle n’est pas prêt.');
  return validateModel(await request('/v1/model', signal));
}
export function batches(rows: SensorRow[], info: ModelInfo): SensorRow[][] {
  const grouped: SensorRow[][] = [];
  const encoder = new TextEncoder();
  let current: SensorRow[] = [],
    bytes = encoder.encode('{"rows":[]}').length;
  for (const row of rows) {
    const size = encoder.encode(JSON.stringify(row)).length;
    if (size + 11 > info.limits.max_request_bytes)
      throw new Error('Une mesure dépasse la limite de corps de cette API.');
    if (
      current.length &&
      (current.length >= info.limits.max_batch_rows ||
        bytes + size + 1 > info.limits.max_request_bytes)
    ) {
      grouped.push(current);
      current = [];
      bytes = 11;
    }
    bytes += size + (current.length ? 1 : 0);
    current.push(row);
  }
  if (current.length) grouped.push(current);
  return grouped;
}
export async function predict(
  rows: SensorRow[],
  info: ModelInfo,
  signal: AbortSignal,
  progress: (done: number) => void,
): Promise<Result> {
  const latest = await connect(signal);
  if (
    identityKey(latest.model) !== identityKey(info.model) ||
    JSON.stringify(latest.feature_names) !== JSON.stringify(info.feature_names)
  ) {
    throw new Error(
      'Le modèle ou son schéma a changé. Reconnectez avant d’importer à nouveau le CSV.',
    );
  }
  const predictions: Result['predictions'] = [];
  for (const group of batches(rows, latest)) {
    const result = validateResult(
      await request('/v1/predict-batch', signal, { rows: group }),
      info.model,
      group,
    );
    const offset = predictions.length;
    predictions.push(...result.predictions.map((p) => ({ ...p, row_index: p.row_index + offset })));
    progress(predictions.length);
  }
  if (signal.aborted) throw new DOMException('Aborted', 'AbortError');
  return { model: info.model, rows: rows.length, predictions };
}
