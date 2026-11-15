import {
  WarningCircle, ArrowLeft, Clock, CheckCircle, HelpCircle,
  Code, Fingerprint, Globe, InfoCircle, Link, Lock,
  ShieldAlert, ShieldCheck, Sparks, Text,
} from 'iconoir-react';
import { useTranslation } from 'react-i18next';
import type { AnalysisResult } from '../lib/analysis';
import { getURLFacts } from '../lib/analysis';

type ReportProps = { result: AnalysisResult; onBack: () => void };

export default function Report({ result, onBack }: ReportProps) {
  const { t, i18n } = useTranslation();
  const facts = getURLFacts(result.url);
  const isPhishing = result.classification === 'potential_phishing';
  const isPending = result.classification === 'not_analyzed';
  const isSample = result.origin === 'sample';
  const displayScore = result.phishingScore === null ? null : Math.round(result.phishingScore * 100);
  const locale = i18n.resolvedLanguage === 'es' ? 'es-ES' : 'en-US';
  const analyzedAt = new Intl.DateTimeFormat(locale, {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(result.analyzedAt);

  const statusTitle = t(isPending
    ? 'report.status.pendingTitle'
    : isPhishing ? 'report.status.phishingTitle' : 'report.status.notFlaggedTitle');
  const statusText = t(isPending
    ? 'report.status.pendingText'
    : isSample ? 'report.status.sampleText'
    : isPhishing ? 'report.status.phishingText' : 'report.status.notFlaggedText');
  const scoreDetail = t(isSample
    ? 'report.score.notReal'
    : isPending ? 'report.score.pending'
    : result.calibrated ? 'report.score.calibrated' : 'report.score.uncalibrated');
  const classificationText = t(isPending
    ? 'report.details.pending'
    : isPhishing ? 'report.details.phishing' : 'report.details.notFlagged');
  const modelVersion = result.origin === 'preview'
    ? t('report.details.modelNotConnected')
    : isSample ? t('report.details.demoOnly') : result.modelVersion;

  return (
    <main className="report-screen">
      <div className="report-topline">
        <button className="back-link" type="button" onClick={onBack}>
          <ArrowLeft width={19} height={19} strokeWidth={1.9} /> {t('report.back')}
        </button>
        <span className="date-pill"><Clock width={15} height={15} /> {analyzedAt}</span>
      </div>

      <div className="report-heading">
        <div className="report-heading__left">
          <h1>{t('report.title')}</h1>
          <p>{result.url}</p>
        </div>
        {isSample && <span className="demo-tag"><Sparks width={14} height={14} /> {t('report.tagSample')}</span>}
        {result.origin === 'preview' && <span className="demo-tag"><InfoCircle width={14} height={14} /> {t('report.tagPreview')}</span>}
        {result.origin === 'api' && <span className="demo-tag"><Lock width={14} height={14} /> {t('report.tagApi')}</span>}
      </div>

      <section className={`verdict-card ${isPhishing ? 'verdict-card--danger' : isPending ? 'verdict-card--neutral' : 'verdict-card--success'}`} aria-label={t('report.summary')}>
        <div className="verdict-card__icon">
          {isPending ? <HelpCircle width={38} height={38} strokeWidth={1.8} /> : isPhishing ? <WarningCircle width={45} height={45} strokeWidth={1.9} /> : <CheckCircle width={45} height={45} strokeWidth={1.8} />}
        </div>
        <div className="verdict-card__copy">
          <h2>{statusTitle}</h2>
          <p>{statusText}</p>
        </div>
        <div className="verdict-card__score">
          <div className="score-label">{t(isSample ? 'report.score.sample' : 'report.score.model')}</div>
          <strong>{displayScore === null ? '—' : `${displayScore}%`}</strong>
          <div className="score-meter" aria-hidden="true"><span style={{ width: `${displayScore ?? 0}%` }} /></div>
          <span className="score-detail">{scoreDetail}</span>
        </div>
      </section>

      <div className="report-grid">
        <section className="glass-panel details-panel" aria-labelledby="details-title">
          <h2 id="details-title">{t('report.details.title')}</h2>
          <dl className="details-table">
            <div><dt>{t('report.details.classification')}</dt><dd><span className={`classification-badge ${isPhishing ? 'classification-badge--danger' : isPending ? 'classification-badge--neutral' : 'classification-badge--success'}`}>{classificationText}</span></dd></div>
            <div><dt>{t('report.details.domain')}</dt><dd title={facts.hostname}>{facts.hostname}</dd></div>
            <div><dt>{t('report.details.protocol')}</dt><dd>{facts.protocol}</dd></div>
            <div><dt>{t('report.details.urlLength')}</dt><dd>{t('report.details.characters', { value: facts.length })}</dd></div>
            <div><dt>{t('report.details.segments')}</dt><dd>{facts.hostnameLabels}</dd></div>
            <div><dt>{t('report.details.model')}</dt><dd>{modelVersion}</dd></div>
          </dl>
        </section>

        <section className="glass-panel signals-panel" aria-labelledby="signals-title">
          <h2 id="signals-title">{t('report.signals.title')}</h2>
          {result.signals.length ? (
            <ul className="signal-list">
              {result.signals.map((signal, index) => {
                const Icon = [Fingerprint, Link, Text, Code][index % 4];
                return <li key={`${signal}-${index}`}><Icon width={23} height={23} strokeWidth={1.75} /><span>{isSample ? t(signal) : signal}</span></li>;
              })}
            </ul>
          ) : (
            <div className="signals-placeholder">
              <Globe width={33} height={33} strokeWidth={1.5} />
              <p>{t('report.signals.empty')}</p>
              <span>{t(isPending ? 'report.signals.pending' : 'report.signals.notAvailable')}</span>
            </div>
          )}
        </section>
      </div>

      <section className="glass-panel contribution-panel" aria-labelledby="contribution-title">
        <div className="contribution-panel__heading">
          <h2 id="contribution-title">{t(isSample ? 'report.contributions.titleSample' : 'report.contributions.titleModel')}</h2>
          <span className="info-indicator" title={t('report.contributions.info')}><InfoCircle width={17} height={17} /></span>
        </div>
        {result.contributions.length ? (
          <div className="contribution-list">
            {result.contributions.map(({ label, value }) => (
              <div className="contribution-row" key={label}>
                <span>{isSample ? t(label) : label}</span>
                <div className="contribution-track" aria-label={`${isSample ? t(label) : label}: ${value}%`}><span style={{ width: `${value}%` }} /></div>
                <strong>{value}%</strong>
              </div>
            ))}
          </div>
        ) : (
          <div className="contribution-empty">
            <ShieldCheck width={22} height={22} strokeWidth={1.7} />
            <p>{t('report.contributions.empty')}</p>
          </div>
        )}
      </section>

      <div className="report-footnote">
        <ShieldAlert width={17} height={17} />
        <span>{t(isSample ? 'report.footnote.sample' : isPending ? 'report.footnote.pending' : 'report.footnote.api')}</span>
      </div>
    </main>
  );
}
