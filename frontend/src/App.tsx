import { useEffect, useRef, useState } from 'react';
import { connect, predict } from './lib/api';
import { exampleRows, FILE_BYTES, MAX_ROWS, parseSensors, sensorsCsv } from './lib/csv';
import type { ModelInfo, Result, SensorRow } from './lib/contracts';
import Results from './Results';
import Reference from './Reference';
import { count, percent, score } from './format';

type View = 'inference' | 'diagnostics' | 'experiments';
const views = [
  {
    id: 'inference' as const,
    label: 'Revue des alertes',
    icon: '01',
    heading: 'Du signal à la décision.',
    intro: 'Importez les mesures, appliquez la décision figée et examinez les alertes APS.',
  },
  {
    id: 'diagnostics' as const,
    label: 'Diagnostic du modèle',
    icon: '02',
    heading: 'Comprendre le compromis.',
    intro: 'Erreurs, calibration et qualité des mesures sur les partitions de référence.',
  },
  {
    id: 'experiments' as const,
    label: 'Expériences',
    icon: '03',
    heading: 'Suivre les preuves.',
    intro: 'Comparez les candidats et inspectez le protocole qui a conduit au modèle figé.',
  },
];
export default function App() {
  const [view, setView] = useState<View>('inference');
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [busy, setBusy] = useState(false);
  const [connecting, setConnecting] = useState(true);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [rows, setRows] = useState<SensorRow[]>([]);
  const [fileName, setFileName] = useState('');
  const [example, setExample] = useState(false);
  const [result, setResult] = useState<Result | null>(null);
  const [reviewed, setReviewed] = useState<Set<number>>(new Set());
  const [done, setDone] = useState(0);
  const controller = useRef<AbortController | null>(null);
  const token = useRef(0);
  const fileInput = useRef<HTMLInputElement>(null);
  const heading = useRef<HTMLHeadingElement>(null);

  async function reconnect() {
    controller.current?.abort();
    const current = ++token.current;
    const active = new AbortController();
    controller.current = active;
    setConnecting(true);
    setInfo(null);
    setRows([]);
    setResult(null);
    setReviewed(new Set());
    setFileName('');
    setError('');
    setNotice('');
    try {
      const model = await connect(active.signal);
      if (token.current === current) setInfo(model);
    } catch (e) {
      if (!active.signal.aborted && token.current === current) setError(message(e));
    } finally {
      if (token.current === current) setConnecting(false);
    }
  }
  useEffect(() => {
    void reconnect();
    return () => {
      ++token.current;
      controller.current?.abort();
    };
  }, []);

  async function loadFile(file: File) {
    if (!info || busy || connecting) return;
    const current = ++token.current;
    setError('');
    setNotice('');
    setRows([]);
    setResult(null);
    setReviewed(new Set());
    setFileName('');
    setExample(false);
    try {
      if (file.size > FILE_BYTES) throw new Error('CSV limité à 5 MiB.');
      const text = await file.text();
      if (token.current !== current) return;
      const values = parseSensors(text, info);
      setRows(values);
      setFileName(file.name);
    } catch (e) {
      if (token.current === current) setError(message(e));
    }
  }
  function useExample() {
    if (!info) return;
    ++token.current;
    setRows(exampleRows(info));
    setFileName('Exemple artificiel · 3 mesures');
    setExample(true);
    setResult(null);
    setReviewed(new Set());
    setError('');
    setNotice('');
  }
  async function analyze() {
    if (!info || !rows.length || busy) return;
    const current = ++token.current;
    const active = new AbortController();
    controller.current = active;
    setBusy(true);
    setDone(0);
    setError('');
    setNotice('');
    setResult(null);
    setReviewed(new Set());
    try {
      const values = await predict(rows, info, active.signal, (n) => {
        if (token.current === current) setDone(n);
      });
      if (token.current === current) setResult(values);
    } catch (e) {
      if (token.current === current) {
        if (active.signal.aborted)
          setNotice('Analyse annulée. Aucun résultat partiel n’est affiché.');
        else setError(message(e));
      }
    } finally {
      if (token.current === current) setBusy(false);
    }
  }
  function downloadExample() {
    if (!info) return;
    const href = URL.createObjectURL(
      new Blob([sensorsCsv(info, exampleRows(info))], { type: 'text/csv;charset=utf-8' }),
    );
    const a = document.createElement('a');
    a.href = href;
    a.download = 'fleetguard-contract-example.csv';
    a.click();
    URL.revokeObjectURL(href);
  }
  const activeView = views.find((v) => v.id === view)!;
  const alerts = result?.predictions.filter((p) => p.predicted_label === 'pos').length;
  const missing = result
    ? result.predictions.reduce((sum, p) => sum + p.missing_fraction, 0) / result.rows
    : null;
  return (
    <div className="app">
      <a className="skip" href="#main">
        Aller au contenu
      </a>
      <aside className="sidebar">
        <a className="brand" href="/" aria-label="FleetGuard accueil">
          <span className="brand-mark">F</span>
          <span>
            FleetGuard<small>APS intelligence</small>
          </span>
        </a>
        <div className="workspace">
          <span className="dot" />
          Espace local<span className="workspace-badge">DEV</span>
        </div>
        <p className="nav-label">EXPLORER</p>
        <nav aria-label="Navigation principale">
          {views.map((v) => (
            <button
              key={v.id}
              aria-current={view === v.id ? 'page' : undefined}
              onClick={() => {
                setView(v.id);
                requestAnimationFrame(() => heading.current?.focus());
              }}
            >
              <span className="nav-icon">{v.icon}</span>
              {v.label}
              <span className="nav-arrow">↗</span>
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <span className="sidebar-symbol">▥</span>
          <strong>Une décision traçable.</strong>
          <p>Pipeline, calibration et seuil conservés ensemble.</p>
          <span className="sidebar-tag">MODÈLE FIGÉ · CPU</span>
          <small>
            Diagnostic historique APS.
            <br />
            Pas une prévision de panne.
          </small>
        </div>
      </aside>
      <div className="workspace-main">
        <header className="topbar">
          <span>
            FleetGuard <span className="slash">/</span> {activeView.label}
          </span>
          <span className={`service-status ${info ? 'online' : ''}`}>
            <span className="dot" />
            {connecting ? 'Connexion…' : info ? 'Service prêt' : 'Service indisponible'}
          </span>
        </header>
        <main id="main">
          <div className="page-heading">
            <div>
              <p className="eyebrow">FLEET WORKSPACE / {activeView.icon}</p>
              <h1 ref={heading} tabIndex={-1}>
                {activeView.heading}
              </h1>
              <p>{activeView.intro}</p>
            </div>
            <span className="page-tag">
              {view === 'inference' ? 'INFÉRENCE LOCALE' : 'RÉFÉRENCE PUBLIÉE'}
            </span>
          </div>
          {error && (
            <div className="notice error" role="alert">
              <strong>Action interrompue</strong>
              <p>{error}</p>
              {!info && (
                <button
                  className="button secondary"
                  disabled={connecting}
                  onClick={() => void reconnect()}
                >
                  Reconnecter le service
                </button>
              )}
            </div>
          )}
          {notice && (
            <div className="notice" role="status">
              {notice}
            </div>
          )}
          {view === 'inference' ? (
            <>
              <div className="two-col input-layout">
                <section className="panel import-panel" aria-labelledby="import-title">
                  <p className="eyebrow">01 / PRÉPARER LES MESURES</p>
                  <h2 id="import-title">Un CSV, une décision figée.</h2>
                  <p className="subtle">
                    {info
                      ? `${info.feature_names.length} capteurs numériques. `
                      : 'Connectez l’API pour obtenir le schéma. '}
                    Sans labels ni identifiants ; cellules vides, na ou null acceptés.
                  </p>
                  <div
                    className={`dropzone ${busy || !info ? 'disabled' : ''}`}
                    onDragOver={(e) => e.preventDefault()}
                    onDrop={(e) => {
                      e.preventDefault();
                      const file = e.dataTransfer.files[0];
                      if (file) void loadFile(file);
                    }}
                  >
                    <span className="upload-mark" aria-hidden="true">
                      ↑
                    </span>
                    <strong>Déposez votre fichier de mesures</strong>
                    <span>
                      CSV séparé par des virgules · {count(MAX_ROWS)} lignes · 5 MiB maximum
                    </span>
                    <input
                      className="sr-only"
                      ref={fileInput}
                      type="file"
                      accept=".csv,text/csv"
                      aria-label="Importer un CSV"
                      disabled={!info || busy || connecting}
                      onChange={(e) => {
                        const file = e.target.files?.[0];
                        if (file) void loadFile(file);
                        e.target.value = '';
                      }}
                    />
                    <button
                      className="button primary"
                      disabled={!info || busy || connecting}
                      onClick={() => fileInput.current?.click()}
                    >
                      Choisir un CSV <span aria-hidden="true">↗</span>
                    </button>
                  </div>
                  <div className="example-tools">
                    <button
                      className="text-button"
                      disabled={!info || busy || connecting}
                      onClick={useExample}
                    >
                      Essayer un exemple
                    </button>
                    <button
                      className="text-button subtle"
                      disabled={!info || busy || connecting}
                      onClick={downloadExample}
                    >
                      Télécharger l’exemple ↓
                    </button>
                  </div>
                  {rows.length > 0 && (
                    <div className="file-ready">
                      <span className="file-icon" aria-hidden="true">
                        CSV
                      </span>
                      <div>
                        <strong>{fileName}</strong>
                        <span>
                          {count(rows.length)} mesures · schéma vérifié dans le navigateur
                        </span>
                      </div>
                      <button
                        className="row-link"
                        disabled={busy}
                        onClick={() => {
                          ++token.current;
                          setRows([]);
                          setResult(null);
                          setReviewed(new Set());
                          setFileName('');
                          setNotice('');
                          setError('');
                        }}
                        aria-label="Retirer le fichier"
                      >
                        ×
                      </button>
                    </div>
                  )}
                  <div className="analyze-tools">
                    <button
                      className="button primary analyze"
                      disabled={!rows.length || busy || !info}
                      onClick={() => void analyze()}
                    >
                      {busy ? 'Analyse en cours…' : 'Analyser les mesures'}{' '}
                      <span aria-hidden="true">→</span>
                    </button>
                    {busy && (
                      <button
                        className="button secondary"
                        onClick={() => controller.current?.abort()}
                      >
                        Annuler
                      </button>
                    )}
                  </div>
                  {busy && (
                    <div className="progress" role="status">
                      <progress max={rows.length} value={done} aria-label="Mesures traitées" />
                      <span>
                        {count(done)} / {count(rows.length)} mesures · lots séquentiels
                      </span>
                    </div>
                  )}
                  {example && rows.length > 0 && (
                    <p className="example-note">
                      <strong>Exemple artificiel.</strong> Mesures null/zero pour tester le
                      parcours, sans interprétation de camion ni mesure de performance.
                    </p>
                  )}
                </section>
                <section className="panel model-panel" aria-labelledby="model-title">
                  <div className="model-top">
                    <p className="eyebrow">02 / MODÈLE ACTIF</p>
                    <span className={`pill ${info ? 'ready' : 'neutral'}`}>
                      {connecting ? 'Connexion' : info ? 'Figé' : 'Hors ligne'}
                    </span>
                  </div>
                  <h2 id="model-title">La décision est versionnée.</h2>
                  {info ? (
                    <>
                      <p className="model-name">
                        {info.model.name}
                        <span>
                          {info.model.dataset_kind === 'official_aps'
                            ? 'Release APS officielle'
                            : 'Fixture synthétique · logiciel seulement'}
                        </span>
                      </p>
                      <dl className="model-facts">
                        <div>
                          <dt>Release active</dt>
                          <dd className="mono">{info.model.release_id}</dd>
                        </div>
                        <div>
                          <dt>Seuil de décision</dt>
                          <dd className="mono">{score(info.model.threshold)}</dd>
                        </div>
                        <div>
                          <dt>Calibration</dt>
                          <dd>
                            {info.model.calibration === 'raw' ? 'Score brut (raw)' : 'Sigmoïde'}
                          </dd>
                        </div>
                        <div>
                          <dt>Politique</dt>
                          <dd>Coût du challenge</dd>
                        </div>
                        <div>
                          <dt>Service</dt>
                          <dd>{info.service_version} · contrat v1</dd>
                        </div>
                      </dl>
                      <div className="model-rule">
                        <span>Score APS ≥ seuil</span>
                        <strong>Décision pos · à inspecter ↗</strong>
                      </div>
                      <details className="identity-details">
                        <summary>Empreintes et limites du modèle</summary>
                        <dl>
                          <dt>Pipeline SHA-256</dt>
                          <dd>{info.model.pipeline_sha256}</dd>
                          <dt>Freeze SHA-256</dt>
                          <dd>{info.model.freeze_sha256}</dd>
                          <dt>Limites par requête</dt>
                          <dd>
                            {info.limits.max_batch_rows} lignes ·{' '}
                            {count(info.limits.max_request_bytes)} octets
                          </dd>
                          <dt>Coût hypothétique</dt>
                          <dd>
                            FP {info.costs.false_positive_cost} / FN{' '}
                            {info.costs.false_negative_cost} unités
                          </dd>
                          <dt>Sémantique</dt>
                          <dd>{info.model.score_semantics}</dd>
                        </dl>
                      </details>
                    </>
                  ) : (
                    <div className="model-empty">
                      <span className="empty-symbol">◇</span>
                      <p>
                        {connecting
                          ? 'Vérification de la disponibilité du modèle…'
                          : 'Démarrez le service local puis reconnectez pour importer vos mesures.'}
                      </p>
                      <button
                        className="button secondary"
                        disabled={connecting}
                        onClick={() => void reconnect()}
                      >
                        Reconnecter le service
                      </button>
                    </div>
                  )}
                  <p className="footnote">
                    Le service conserve le prétraitement, la calibration et le seuil. Votre import
                    ne les ajuste pas.
                  </p>
                </section>
              </div>
              <div className="stats">
                <Stat
                  label="Mesures analysées"
                  value={result ? count(result.rows) : '—'}
                  note="Positions dans le fichier importé"
                />
                <Stat
                  label="Décisions à inspecter"
                  value={alerts === undefined ? '—' : count(alerts)}
                  note="Score APS supérieur ou égal au seuil"
                  alert
                />
                <Stat
                  label="Part signalée du lot"
                  value={result && alerts !== undefined ? percent(alerts / result.rows) : '—'}
                  note="Volume d’alertes, pas une précision"
                />
                <Stat
                  label="Mesures absentes"
                  value={missing === null ? '—' : percent(missing)}
                  note="Moyenne des fractions par ligne"
                />
              </div>
              {result ? (
                <Results
                  key={fileName + result.model.release_id + done}
                  result={result}
                  input={rows}
                  reviewed={reviewed}
                  setReviewed={setReviewed}
                />
              ) : (
                <section className="empty-state">
                  <span className="empty-symbol">◎</span>
                  <div>
                    <h2>Votre revue commence ici.</h2>
                    <p>
                      Importez un CSV ou essayez l’exemple, puis lancez l’analyse. Les décisions et
                      les capteurs seront inspectables ligne par ligne.
                    </p>
                  </div>
                </section>
              )}
              <div className="privacy-note">
                <span aria-hidden="true">◇</span>
                <p>
                  Le CSV est lu dans votre navigateur ; seuls les capteurs sont envoyés à votre API
                  pour l’inférence. Aucun fichier n’est sauvegardé par ce parcours. Les annotations
                  restent dans cette session, sauf export volontaire.
                </p>
              </div>
            </>
          ) : (
            <Reference key={view} view={view} />
          )}
        </main>
        <footer>
          FleetGuard <span>Diagnostic APS · données anonymisées · décision traçable</span>
          <span>v0.6</span>
        </footer>
      </div>
    </div>
  );
}
function message(e: unknown): string {
  return e instanceof Error ? e.message : 'L’opération a échoué.';
}
function Stat({
  label,
  value,
  note,
  alert = false,
}: {
  label: string;
  value: string;
  note: string;
  alert?: boolean;
}) {
  return (
    <article className={`stat ${alert ? 'alert-stat' : ''}`}>
      <p>{label}</p>
      <strong>{value}</strong>
      <small>{note}</small>
    </article>
  );
}
