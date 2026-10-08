import { describe, expect, it } from 'vitest';
import { exampleRows, MAX_ROWS, parseSensors, sensorsCsv } from './csv';
import { model } from './fixtures.test-helper';
describe('sensor-only CSV contract', () => {
  it('preserves input order, reordered features, quoted headers, BOM, exponents and explicit missing values', () => {
    expect(parseSensors('\uFEFF"b",a\r\nna,1e-3\r\n3,-2.5', model)).toEqual([
      { b: null, a: 0.001 },
      { b: 3, a: -2.5 },
    ]);
  });
  it('retains all-empty sensor rows while ignoring truly blank lines', () => {
    expect(parseSensors('a,b\n,\n\nnull,na\n', model)).toEqual([
      { a: null, b: null },
      { a: null, b: null },
    ]);
  });
  it.each(['true', 'NaN', 'Infinity', '1e100', '=SUM(1)', '0x10', '1 000', '12kg'])(
    'rejects ambiguous/nonfinite/out-of-range values: %s',
    (value) => {
      expect(() => parseSensors(`a,b\n${value},2`, model)).toThrow('nombre fini');
    },
  );
  it.each(['a,a\n1,2', 'a,class\n1,pos', 'a\n1', 'a,b,c\n1,2,3', 'a,b\n1,2,3', 'a;b\n1;2'])(
    'rejects wrong schema or row width: %s',
    (csv) => {
      expect(() => parseSensors(csv, model)).toThrow();
    },
  );
  it('rejects empty and malformed input', () => {
    expect(() => parseSensors('a,b\n', model)).toThrow('au moins');
    expect(() => parseSensors('a,b\n"unclosed,2', model)).toThrow('mal formé');
  });
  it('bounds browser parsing by bytes and rows', () => {
    expect(() => parseSensors('a'.repeat(5 * 1024 * 1024 + 1), model)).toThrow('5 MiB');
    expect(() => parseSensors(`a,b\n${'1,2\n'.repeat(MAX_ROWS + 1)}`, model)).toThrow('5000');
  });
  it('generates an explicitly artificial downloadable example against the active schema', () => {
    const rows = exampleRows(model);
    expect(rows).toHaveLength(3);
    expect(parseSensors(sensorsCsv(model, rows), model)).toEqual(rows);
  });
});
