import { useEffect, useState } from 'react';
import { count, percent, score } from './format';

type Metrics = {
  rows: number;
  fp: number;
  fn: number;
  tp: number;
  tn: number;
  recall: number;
  precision: number;
  cost: number;
  brier_score: number;
  average_precision: number;
};
type Experiment = {
  name: string;
  rows: number;
  cost: number;
  cost_per_row: number;
  recall: number;
  precision: number;
  average_precision: number;
};
type Evidence = {
  release_id: string;
  freeze_sha256: string;
  calibration: string;
  threshold: number;
  official_test: Metrics;
  validation: { release: Metrics; source_raw: Metrics };
  missingness: {
    lower: number;
    upper: number;
    rows: number;
    positives: number;
    fp: number;
    fn: number;
    recall: number | null;
  }[];
  permutation: { name: string; columns: number; cost_delta: number; std: number }[];
  comparison: Experiment[];
  optimization: Experiment[];
  sources: { path: string; sha256: string }[];
};
export default function Reference({ view }: { view: 'diagnostics' | 'experiments' }) {
  const [data, setData] = useState<Evidence | null>(null);
  const [error, setError] = useState(false);
  const [stage, setStage] = useState('optimization');
  useEffect(() => {
    const controller = new AbortController();
    fetch('/evidence/reference.json', { signal: controller.signal })
      .then((r) => {
        if (!r.ok) throw new Error('Reference unavailable');
        return r.json();
      })
      .then((d) => setData(d))
      .catch((e) => {
        if (e.name !== 'AbortError') setError(true);
      });
    return () => controller.abort();
  }, []);
  if (error)
    return (
      <div className="notice error" role="alert">
        Les preuves de référence sont indisponibles. Relancez npm run build pour synchroniser les
        agrégats versionnés.
      </div>
    );
  if (!data) return <p role="status">Chargement des résultats de référence…</p>;
  const m = data.official_test;
  const experiments = stage === 'optimization' ? data.optimization : data.comparison;
  return (
    <>
      <div className="notice reference">
        <strong>Référence publiée · pas le lot importé</strong>
        <p>
          Ces chiffres décrivent la release <code>{data.release_id}</code> sur le snapshot Scania.
          Ils ne sont pas attribués au modèle actif, même si son nom est identique.
        </p>
      </div>
      {view === 'diagnostics' ? (
        <>
          <div className="stats">
            <Stat
              label="Rappel APS · test officiel"
              value={percent(m.recall)}
              note={`${m.tp} positifs détectés / ${m.tp + m.fn}`}
            />
            <Stat
              label="Précision · test officiel"
              value={percent(m.precision)}
              note={`${m.tp} positifs / ${m.tp + m.fp} alertes`}
            />
            <Stat
              label="Coût du challenge"
              value={count(m.cost)}
              note={`10 × ${m.fp} FP + 500 × ${m.fn} FN`}
            />
            <Stat
              label="Average precision"
              value={score(m.average_precision)}
              note={`${count(m.rows)} observations · décision figée`}
            />
          </div>
          <div className="two-col">
            <section className="panel">
              <p className="eyebrow">ÉVALUATION FINALE UNIQUE</p>
              <h2>Ce que la décision manque.</h2>
              <p className="subtle">
                Seuil {score(data.threshold)} · {data.calibration}. Aucun ajustement après le test.
              </p>
              <table className="matrix">
                <caption>Matrice de confusion · test officiel</caption>
                <thead>
                  <tr>
                    <th>Label réel</th>
                    <th>Prédit neg</th>
                    <th>Prédit pos</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <th>neg</th>
                    <td>
                      {count(m.tn)}
                      <small>Vrais négatifs</small>
                    </td>
                    <td className="fp">
                      {count(m.fp)}
                      <small>Fausses alertes</small>
                    </td>
                  </tr>
                  <tr>
                    <th>pos</th>
                    <td className="fn">
                      {m.fn}
                      <small>APS manqués</small>
                    </td>
                    <td>
                      {m.tp}
                      <small>APS détectés</small>
                    </td>
                  </tr>
                </tbody>
              </table>
              <p className="footnote">
                Le coût est exprimé en unités du challenge, pas en euros. Les labels neg décrivent
                des pannes hors APS.
              </p>
            </section>
            <section className="panel">
              <p className="eyebrow">QUALITÉ DU SCORE · VALIDATION</p>
              <h2>Calibration et décision.</h2>
              <p className="subtle">
                Brier plus faible après calibration sur le rôle réservé. Les décisions de validation
                sont identiques.
              </p>
              <div className="comparison-stat">
                <span>Source brute</span>
                <strong>{score(data.validation.source_raw.brier_score)}</strong>
              </div>
              <div className="comparison-stat">
                <span>Release sigmoïde</span>
                <strong>{score(data.validation.release.brier_score)}</strong>
              </div>
              <div className="callout">
                {data.validation.release.fp} FP et {data.validation.release.fn} FN avant/après.
                Améliorer l’échelle ne garantit pas une probabilité fiable sur une autre flotte.
              </div>
            </section>
          </div>
          <section className="panel">
            <p className="eyebrow">MANQUE DE MESURES · VALIDATION</p>
            <h2>Chaque tranche a son dénominateur.</h2>
            <div className="table-scroll">
              <table>
                <caption className="sr-only">
                  Erreurs de validation par fraction de capteurs absents
                </caption>
                <thead>
                  <tr>
                    <th>Capteurs absents</th>
                    <th>Lignes</th>
                    <th>Positifs réels</th>
                    <th>FP</th>
                    <th>FN</th>
                    <th>Rappel APS</th>
                  </tr>
                </thead>
                <tbody>
                  {data.missingness.map((r) => (
                    <tr key={r.lower}>
                      <td>
                        {percent(r.lower)}–{percent(r.upper)}
                      </td>
                      <td>{count(r.rows)}</td>
                      <td>{r.positives}</td>
                      <td>{r.fp}</td>
                      <td>{r.fn}</td>
                      <td>
                        {r.recall === null ? 'Non défini · aucun positif' : percent(r.recall)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="footnote">
              Tranches [borne basse, borne haute), dernière borne haute incluse. Une tranche sans
              positif ne permet pas d’estimer le rappel.
            </p>
          </section>
          <section className="panel">
            <p className="eyebrow">SENSIBILITÉ · PERMUTATION GROUPÉE</p>
            <h2>Blocs de capteurs et coût.</h2>
            <p className="subtle">
              Cinq plus fortes hausses moyennes du coût par ligne, sur validation et à seuil fixe.
              Trois répétitions ; dispersion ± écart-type.
            </p>
            <div className="bars">
              {data.permutation.map((r) => (
                <div className="bar-row" key={r.name}>
                  <span className="mono">
                    {r.name} · {r.columns} capteurs
                  </span>
                  <div className="bar-track" aria-hidden="true">
                    <i
                      style={{
                        width: `${(Math.max(0, r.cost_delta) / Math.max(...data.permutation.map((p) => p.cost_delta))) * 100}%`,
                      }}
                    />
                  </div>
                  <strong>
                    {score(r.cost_delta)} ± {score(r.std)}
                  </strong>
                </div>
              ))}
            </div>
            <p className="footnote">
              La permutation mesure une sensibilité, pas une cause. Des blocs corrélés et des
              variables anonymisées limitent l’interprétation physique.
            </p>
          </section>
        </>
      ) : (
        <>
          <div className="two-col">
            <section className="panel">
              <p className="eyebrow">PROTOCOLE</p>
              <h2>Une décision, plusieurs rôles.</h2>
              <ol className="protocol">
                <li>
                  <strong>Fit</strong>
                  <span>Apprendre uniquement sur le rôle d’ajustement.</span>
                </li>
                <li>
                  <strong>Calibration</strong>
                  <span>Comparer raw/sigmoid sur son rôle réservé.</span>
                </li>
                <li>
                  <strong>Seuil</strong>
                  <span>Choisir le coût minimum hors du scoring.</span>
                </li>
                <li>
                  <strong>Scoring → freeze → test</strong>
                  <span>Mesurer séparément, puis évaluer une fois.</span>
                </li>
              </ol>
            </section>
            <section className="panel">
              <p className="eyebrow">RECHERCHE BORNÉE</p>
              <h2>Le score de sélection a ses limites.</h2>
              <p>
                Les essais Optuna et le champion réutilisent les folds de développement. Leur
                meilleur coût CV est un score de sélection, avec un risque d’optimisme.
              </p>
              <p>
                Le MLP choisit ses epochs à l’intérieur de fit, avant calibration séparée. Les
                différences entre étapes ont aussi des tailles d’ajustement et protocoles
                différents.
              </p>
              <div className="callout">
                Le test officiel publié reste une évaluation figée. L’interface ne le réutilise
                jamais pour régler un seuil.
              </div>
            </section>
          </div>
          <section className="panel">
            <div className="section-top">
              <div>
                <p className="eyebrow">SCORING CV · 48 000 OBSERVATIONS</p>
                <h2>Comparer dans un même protocole.</h2>
              </div>
              <label className="sort">
                Étape
                <select value={stage} onChange={(e) => setStage(e.target.value)}>
                  <option value="optimization">03 · Optimisation</option>
                  <option value="comparison">02 · Comparaison</option>
                </select>
              </label>
            </div>
            <p className="subtle">
              Trois folds. Le rappel, la précision et l’AP ci-dessous sont des moyennes par fold ;
              le coût agrège les erreurs hors fold.
            </p>
            <div className="table-scroll">
              <table>
                <caption className="sr-only">
                  Résultats des candidats dans l’étape sélectionnée
                </caption>
                <thead>
                  <tr>
                    <th>Candidat</th>
                    <th>Coût / ligne</th>
                    <th>Coût total</th>
                    <th>Rappel moyen</th>
                    <th>Précision moyenne</th>
                    <th>AP moyenne</th>
                  </tr>
                </thead>
                <tbody>
                  {experiments.map((r) => (
                    <tr key={r.name}>
                      <td className="mono">{r.name}</td>
                      <td>{score(r.cost_per_row)}</td>
                      <td>{count(r.cost)}</td>
                      <td>{percent(r.recall)}</td>
                      <td>{percent(r.precision)}</td>
                      <td>{score(r.average_precision)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </>
      )}
      <section className="panel sources">
        <p className="eyebrow">TRACER LES PREUVES</p>
        <h2>Sources versionnées.</h2>
        <p className="subtle">
          Reproduisez les expériences depuis le dépôt. Ces empreintes identifient les agrégats
          intégrés à cette build.
        </p>
        <details>
          <summary>Release, freeze et sources SHA-256</summary>
          <dl>
            <dt>Release de référence</dt>
            <dd>{data.release_id}</dd>
            <dt>Freeze</dt>
            <dd>{data.freeze_sha256}</dd>
            {data.sources.map((s) => (
              <div key={s.path}>
                <dt>{s.path}</dt>
                <dd>{s.sha256}</dd>
              </div>
            ))}
          </dl>
        </details>
        <p className="footnote">
          Benchmark diagnostique historique, sans identifiant camion ni timestamp exploitable. Le
          transfert à une autre flotte et la prévision de pannes futures ne sont pas démontrés.
        </p>
      </section>
    </>
  );
}
function Stat({ label, value, note }: { label: string; value: string; note: string }) {
  return (
    <article className="stat">
      <p>{label}</p>
      <strong>{value}</strong>
      <small>{note}</small>
    </article>
  );
}
