import Papa from 'papaparse';
import type { ModelInfo, SensorRow } from './contracts';

export const FILE_BYTES = 5 * 1024 * 1024;
export const MAX_ROWS = 5000;
const numeric = /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/;

export function parseSensors(text: string, info: ModelInfo): SensorRow[] {
  if (new TextEncoder().encode(text).length > FILE_BYTES) throw new Error('CSV limité à 5 MiB.');
  const parsed = Papa.parse<string[]>(text.replace(/^\uFEFF/, ''), {
    delimiter: ',',
    skipEmptyLines: true,
    dynamicTyping: false,
  });
  if (parsed.errors.length) throw new Error('CSV mal formé : guillemets ou séparateurs invalides.');
  const [header, ...lines] = parsed.data;
  if (!header || !lines.length)
    throw new Error('Le CSV doit contenir un en-tête et au moins une mesure.');
  if (lines.length > MAX_ROWS) throw new Error(`CSV limité à ${MAX_ROWS} lignes de mesures.`);
  if (new Set(header).size !== header.length)
    throw new Error('Des noms de colonnes sont dupliqués.');
  const expected = new Set(info.feature_names);
  const extra = header.filter((n) => !expected.has(n));
  const missing = info.feature_names.filter((n) => !header.includes(n));
  if (extra.length || missing.length)
    throw new Error(
      `Colonnes incompatibles. Manquantes : ${missing.slice(0, 6).join(', ') || 'aucune'}. ` +
        `Supplémentaires : ${extra.slice(0, 6).join(', ') || 'aucune'}. Attendu : ${expected.size} capteurs, sans class ni row_id.`,
    );
  return lines.map((line, i) => {
    if (line.length !== header.length)
      throw new Error(`Mesure ${i + 1} : nombre de champs incorrect.`);
    const row: SensorRow = Object.create(null);
    line.forEach((cell, j) => {
      const value = cell.trim();
      if (value === '' || value === 'na' || value === 'null') row[header[j]] = null;
      else {
        const number = Number(value);
        if (
          !numeric.test(value) ||
          !Number.isFinite(number) ||
          Math.abs(number) > info.limits.feature_abs_max
        ) {
          throw new Error(
            `Mesure ${i + 1}, capteur ${header[j]} : nombre fini compatible float32 ou na attendu.`,
          );
        }
        row[header[j]] = number;
      }
    });
    return row;
  });
}
export function exampleRows(info: ModelInfo): SensorRow[] {
  return [0, 1, 2].map((i) =>
    Object.fromEntries(
      info.feature_names.map((name, j) => [name, i === 0 || (i === 2 && j % 3 === 0) ? null : 0]),
    ),
  );
}
export function sensorsCsv(info: ModelInfo, rows: SensorRow[]): string {
  return Papa.unparse({
    fields: info.feature_names,
    data: rows.map((row) => info.feature_names.map((name) => row[name] ?? 'na')),
  });
}
