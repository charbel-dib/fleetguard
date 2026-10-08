import { useEffect, useMemo, useRef, useState } from 'react';
import type { Dispatch, SetStateAction } from 'react';
import type { Prediction, Result, SensorRow } from './lib/contracts';
import { score, percent } from './format';

export default function Results({
  result,
  input,
  reviewed,
  setReviewed,
}: {
  result: Result;
  input: SensorRow[];
  reviewed: Set<number>;
  setReviewed: Dispatch<SetStateAction<Set<number>>>;
}) {
  const [filter, setFilter] = useState<'all' | 'pos' | 'neg'>('all');
  const [sort, setSort] = useState('score');
  const [page, setPage] = useState(0);
  const [selected, setSelected] = useState<Prediction | null>(null);
  const [query, setQuery] = useState('');
  const dialog = useRef<HTMLDialogElement>(null);
  const visible = useMemo(
    () =>
      result.predictions
        .filter((p) => filter === 'all' || p.predicted_label === filter)
        .sort((a, b) =>
          sort === 'row'
            ? a.row_index - b.row_index
            : b.positive_score - a.positive_score || a.row_index - b.row_index,
        ),
    [result, filter, sort],
  );
  const pages = Math.max(1, Math.ceil(visible.length / 20));
  useEffect(() => {
    if (selected) {
      setQuery('');
      dialog.current?.showModal();
    } else dialog.current?.close();
  }, [selected]);
  const toggle = (index: number) =>
    setReviewed((prev) => {
      const next = new Set(prev);
      if (next.has(index)) next.delete(index);
      else next.add(index);
      return next;
    });
  function download() {
    const blob = new Blob(
      [
        JSON.stringify(
          {
            contract_version: 1,
            model: result.model,
            rows: result.rows,
            predictions: result.predictions.map((p) => ({
              ...p,
              reviewed: reviewed.has(p.row_index),
            })),
          },
          null,
          2,
        ),
      ],
      { type: 'application/json' },
    );
    const href = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = href;
    a.download = 'fleetguard-review.json';
    a.click();
    URL.revokeObjectURL(href);
  }
  return (
    <section className="panel results" aria-labelledby="results-title">
      <div className="section-top">
        <div>
          <p className="eyebrow">REVUE DU LOT</p>
          <h2 id="results-title">Des scores à examiner.</h2>
        </div>
        <button className="button secondary" onClick={download}>
          Exporter la revue ↗
        </button>
      </div>
      <p className="subtle">
        Le seuil est figé. Les observations de ce lot n’ont pas de labels : aucun rappel, coût
        d’erreur ou gain métier n’est calculé ici.
      </p>
      <div className="table-tools">
        <div className="segmented" aria-label="Filtrer les décisions">
          {(
            [
              ['all', 'Toutes'],
              ['pos', 'À inspecter'],
              ['neg', 'Autres composants'],
            ] as const
          ).map(([value, label]) => (
            <button
              key={value}
              aria-pressed={filter === value}
              onClick={() => {
                setFilter(value);
                setPage(0);
              }}
            >
              {label}
            </button>
          ))}
        </div>
        <label className="sort">
          Trier par{' '}
          <select
            value={sort}
            onChange={(e) => {
              setSort(e.target.value);
              setPage(0);
            }}
          >
            <option value="score">Score décroissant</option>
            <option value="row">Ordre du CSV</option>
          </select>
        </label>
      </div>
      <div className="table-scroll">
        <table>
          <caption className="sr-only">Décisions du lot, par position de mesure</caption>
          <thead>
            <tr>
              <th>Mesure</th>
              <th>Décision APS</th>
              <th>Score APS</th>
              <th>Mesures absentes</th>
              <th>Revue locale</th>
              <th>
                <span className="sr-only">Détail</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {visible.slice(page * 20, (page + 1) * 20).map((p) => (
              <tr key={p.row_index}>
                <td className="mono">{String(p.row_index + 1).padStart(3, '0')}</td>
                <td>
                  <span className={`pill ${p.predicted_label === 'pos' ? 'alert' : 'neutral'}`}>
                    {p.predicted_label === 'pos' ? 'À inspecter' : 'Autres composants'}
                  </span>
                </td>
                <td>
                  <span className="mono">{score(p.positive_score)}</span>
                  <div className="score-track" aria-hidden="true">
                    <span style={{ width: `${p.positive_score * 100}%` }} />
                  </div>
                </td>
                <td>{percent(p.missing_fraction)}</td>
                <td>
                  <label className="check">
                    <input
                      type="checkbox"
                      checked={reviewed.has(p.row_index)}
                      onChange={() => toggle(p.row_index)}
                      aria-label={`Mesure ${p.row_index + 1} examinée`}
                    />
                    Examinée
                  </label>
                </td>
                <td>
                  <button
                    className="row-link"
                    aria-label={`Voir mesure ${p.row_index + 1}`}
                    onClick={() => setSelected(p)}
                  >
                    Voir ↗
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {!visible.length && <p className="empty">Aucune mesure ne correspond à ce filtre.</p>}
      <div className="pagination">
        <span>
          {visible.length} mesures · {reviewed.size} examinées dans cette session
        </span>
        <div>
          <button
            disabled={page === 0}
            onClick={() => setPage((p) => p - 1)}
            aria-label="Page précédente"
          >
            ←
          </button>
          <span>
            Page {page + 1} / {pages}
          </span>
          <button
            disabled={page + 1 >= pages}
            onClick={() => setPage((p) => p + 1)}
            aria-label="Page suivante"
          >
            →
          </button>
        </div>
      </div>
      <p className="footnote">
        « Examinée » est une annotation de session, conservée uniquement dans l’export. La mesure
        est une position dans le CSV, pas un identifiant de camion. Une décision neg ne signifie pas
        « camion sain ».
      </p>
      <dialog
        ref={dialog}
        onCancel={() => setSelected(null)}
        onClose={() => setSelected(null)}
        aria-labelledby="detail-title"
      >
        {selected && (
          <>
            <div className="section-top">
              <div>
                <p className="eyebrow">DÉTAIL DE LA MESURE</p>
                <h2 id="detail-title">Mesure {selected.row_index + 1}</h2>
              </div>
              <button className="button secondary" onClick={() => setSelected(null)} autoFocus>
                Fermer
              </button>
            </div>
            <div className="detail-metrics">
              <div>
                <span>Score APS</span>
                <strong>{score(selected.positive_score)}</strong>
              </div>
              <div>
                <span>Seuil figé</span>
                <strong>{score(result.model.threshold)}</strong>
              </div>
              <div>
                <span>Mesures absentes</span>
                <strong>{percent(selected.missing_fraction)}</strong>
              </div>
            </div>
            <p>
              Décision : <strong>{selected.predicted_label}</strong> ·{' '}
              {result.model.calibration === 'raw' ? 'Score brut' : 'Calibration sigmoïde'}. Le score
              ne constitue pas une prévision de panne future.
            </p>
            <p className="footnote">
              Capteurs anonymisés : cette vue expose les entrées, sans explication physique ou
              attribution causale individuelle.
            </p>
            <label className="sensor-search">
              Rechercher un capteur
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Ex. aa_000"
              />
            </label>
            <div className="sensor-grid">
              {Object.entries(input[selected.row_index])
                .filter(([name]) => name.includes(query))
                .map(([name, value]) => (
                  <div key={name}>
                    <code>{name}</code>
                    <span className="mono">
                      {value === null
                        ? 'Absente'
                        : value.toLocaleString('fr-FR', { maximumSignificantDigits: 8 })}
                    </span>
                  </div>
                ))}
            </div>
            <details className="identity-details">
              <summary>Identité utilisée pour cette décision</summary>
              <dl>
                <dt>Release</dt>
                <dd>{result.model.release_id}</dd>
                <dt>Pipeline SHA-256</dt>
                <dd>{result.model.pipeline_sha256}</dd>
                <dt>Freeze SHA-256</dt>
                <dd>{result.model.freeze_sha256}</dd>
              </dl>
            </details>
          </>
        )}
      </dialog>
    </section>
  );
}
