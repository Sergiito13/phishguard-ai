import {
  WarningCircle, ArrowLeft, Clock, CheckCircle, HelpCircle,
  Code, Fingerprint, Globe, InfoCircle, Link, Lock,
  ShieldAlert, ShieldCheck, Sparks, Text,
} from 'iconoir-react';
import type { AnalysisResult } from '../lib/analysis';
import { getURLFacts } from '../lib/analysis';

type ReportProps = { result: AnalysisResult; onBack: () => void };

export default function Report({ result, onBack }: ReportProps) {
  const facts = getURLFacts(result.url);
  const isPhishing = result.classification === 'potential_phishing';
  const isPending = result.classification === 'not_analyzed';
  const displayScore = result.phishingScore === null ? null : Math.round(result.phishingScore * 100);
  const isSample = result.origin === 'sample';

  const statusTitle = isPending
    ? 'Your URL is ready to preview'
    : isPhishing
      ? 'Potential phishing detected'
      : 'No suspicious classification';

  const statusText = isPending
    ? 'The interface is ready. Connect the FastAPI backend to obtain a model prediction.'
    : isSample
      ? 'This is an illustrative sample report. The classification and score are not from the trained model.'
      : isPhishing
        ? 'The experimental model flagged this URL. Review it carefully before visiting.'
        : 'The experimental model did not flag this URL. This does not establish that it is safe.';

  return (
    <main className="report-screen">
      <div className="report-topline">
        <button className="back-link" type="button" onClick={onBack}>
          <ArrowLeft width={19} height={19} strokeWidth={1.9} /> Back to home
        </button>
        <span className="date-pill"><Clock width={15} height={15} /> {result.analyzedAt.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })} · {result.analyzedAt.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}</span>
      </div>

      <div className="report-heading">
        <div className="report-heading__left">
          <h1>Analysis report</h1>
          <p>{result.url}</p>
        </div>
        {isSample && <span className="demo-tag"><Sparks width={14} height={14} /> ILLUSTRATIVE DEMO</span>}
        {result.origin === 'preview' && <span className="demo-tag"><InfoCircle width={14} height={14} /> PREVIEW ONLY</span>}
        {result.origin === 'api' && <span className="demo-tag"><Lock width={14} height={14} /> EXPERIMENTAL ML</span>}
      </div>

      <section className={`verdict-card ${isPhishing ? 'verdict-card--danger' : isPending ? 'verdict-card--neutral' : 'verdict-card--success'}`} aria-label="Analysis summary">
        <div className="verdict-card__icon">
          {isPending ? <HelpCircle width={38} height={38} strokeWidth={1.8} /> : isPhishing ? <WarningCircle width={45} height={45} strokeWidth={1.9} /> : <CheckCircle width={45} height={45} strokeWidth={1.8} />}
        </div>
        <div className="verdict-card__copy">
          <h2>{statusTitle}</h2>
          <p>{statusText}</p>
        </div>
        <div className="verdict-card__score">
          <div className="score-label">{isSample ? 'Demo score' : 'Model score'}</div>
          <strong>{displayScore === null ? '—' : `${displayScore}%`}</strong>
          <div className="score-meter" aria-hidden="true"><span style={{ width: `${displayScore ?? 0}%` }} /></div>
          <span className="score-detail">{isSample ? 'Not a real prediction' : isPending ? 'Awaiting API connection' : result.calibrated ? 'Calibrated by backend' : 'Uncalibrated output'}</span>
        </div>
      </section>

      <div className="report-grid">
        <section className="glass-panel details-panel" aria-labelledby="details-title">
          <h2 id="details-title">Analysis details</h2>
          <dl className="details-table">
            <div><dt>Classification</dt><dd><span className={`classification-badge ${isPhishing ? 'classification-badge--danger' : isPending ? 'classification-badge--neutral' : 'classification-badge--success'}`}>{isPending ? 'Pending model' : isPhishing ? 'Phishing flagged' : 'Not flagged'}</span></dd></div>
            <div><dt>Domain</dt><dd title={facts.hostname}>{facts.hostname}</dd></div>
            <div><dt>Protocol</dt><dd>{facts.protocol}</dd></div>
            <div><dt>URL length</dt><dd>{facts.length} characters</dd></div>
            <div><dt>Hostname segments</dt><dd>{facts.hostnameLabels}</dd></div>
            <div><dt>Model</dt><dd>{result.modelVersion}</dd></div>
          </dl>
        </section>

        <section className="glass-panel signals-panel" aria-labelledby="signals-title">
          <h2 id="signals-title">Key signals detected</h2>
          {result.signals.length ? (
            <ul className="signal-list">
              {result.signals.map((signal, index) => {
                const Icon = [Fingerprint, Link, Text, Code][index % 4];
                return <li key={`${signal}-${index}`}><Icon width={23} height={23} strokeWidth={1.75} /><span>{signal}</span></li>;
              })}
            </ul>
          ) : (
            <div className="signals-placeholder">
              <Globe width={33} height={33} strokeWidth={1.5} />
              <p>No explainability signals are available yet.</p>
              <span>{isPending ? 'Connect the backend to analyze this URL.' : 'The current model does not provide verified per-URL explanations.'}</span>
            </div>
          )}
        </section>
      </div>

      <section className="glass-panel contribution-panel" aria-labelledby="contribution-title">
        <div className="contribution-panel__heading">
          <h2 id="contribution-title">Feature contribution {isSample ? '(sample)' : '(model)'}</h2>
          <span className="info-indicator" title="Explainability values must be supplied by the backend; sample bars are illustrative."><InfoCircle width={17} height={17} /></span>
        </div>
        {result.contributions.length ? (
          <div className="contribution-list">
            {result.contributions.map(({ label, value }) => (
              <div className="contribution-row" key={label}>
                <span>{label}</span>
                <div className="contribution-track" aria-label={`${label}: ${value}%`}><span style={{ width: `${value}%` }} /></div>
                <strong>{value}%</strong>
              </div>
            ))}
          </div>
        ) : (
          <div className="contribution-empty">
            <ShieldCheck width={22} height={22} strokeWidth={1.7} />
            <p>Feature-level explanations will appear when supported by a validated model. We do not invent attribution percentages.</p>
          </div>
        )}
      </section>

      <div className="report-footnote">
        <ShieldAlert width={17} height={17} />
        <span>{isSample ? 'Illustrative screen only. The 94% score and feature breakdown are placeholders, not ML output.' : isPending ? 'Frontend-only preview: no model inference has taken place.' : 'Experimental model: an unflagged URL may still be malicious. Scores are not verified probabilities of phishing.'}</span>
      </div>
    </main>
  );
}
