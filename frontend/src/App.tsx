import { useEffect, useState } from 'react';
import { Navigate, Route, Routes, useLocation, useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { ArrowRight, CheckCircle, Database, Search, ShieldCheck, Xmark } from 'iconoir-react';
import Header from './components/Header';
import BackgroundArt from './components/BackgroundArt';
import Home from './components/Home';
import Report from './components/Report';
import {
  AnalysisError,
  analyzeURL,
  createSampleResult,
  type AnalysisFailure,
  type AnalysisResult,
} from './lib/analysis';

type Theme = 'dark' | 'light';
type RouteState = { result?: AnalysisResult } | null;

export default function App() {
  const { t } = useTranslation();
  const [theme, setTheme] = useState<Theme>('dark');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<AnalysisFailure | null>(null);
  const [howOpen, setHowOpen] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();

  const goHome = () => {
    setError(null);
    navigate('/');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const onAnalyze = async (url: string) => {
    setError(null);
    setBusy(true);
    try {
      const result = await analyzeURL(url);
      // Avoid putting potentially sensitive URL parameters in browser history.
      navigate('/analysis', { state: { result } });
      window.scrollTo({ top: 0, behavior: 'smooth' });
    } catch (reason) {
      setError(reason instanceof AnalysisError
        ? { key: reason.key, options: reason.options }
        : { key: 'errors.unexpected' });
    } finally {
      setBusy(false);
    }
  };

  const onSample = () => {
    navigate('/analysis?sample=1', { state: { result: createSampleResult() } });
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setHowOpen(false);
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, []);

  const params = new URLSearchParams(location.search);
  const rememberedResult = (location.state as RouteState)?.result;
  let result: AnalysisResult | null = rememberedResult ?? null;
  if (!result && params.get('sample') === '1') result = createSampleResult();

  return (
    <div className="app-root" data-theme={theme}>
      <div className="app-shell">
        <BackgroundArt />
        <Header
          theme={theme}
          onThemeToggle={() => setTheme(current => current === 'dark' ? 'light' : 'dark')}
          onHome={goHome}
          onHow={() => setHowOpen(true)}
        />
        <Routes>
          <Route path="/" element={<Home onAnalyze={onAnalyze} onSample={onSample} busy={busy} error={error} />} />
          <Route path="/analysis" element={result ? <Report result={result} onBack={goHome} /> : <Navigate to="/" replace />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </div>

      {howOpen && (
        <div
          className="modal-overlay"
          role="presentation"
          onMouseDown={event => { if (event.target === event.currentTarget) setHowOpen(false); }}
        >
          <section className="how-modal" role="dialog" aria-modal="true" aria-labelledby="how-title">
            <button type="button" className="modal-close" onClick={() => setHowOpen(false)} aria-label={t('how.close')}>
              <Xmark width={21} height={21} />
            </button>
            <span className="modal-kicker">{t('how.kicker')}</span>
            <h2 id="how-title">{t('how.titleFirst')}<br /><span>{t('how.titleAccent')}</span></h2>
            <p className="modal-lead">{t('how.lead')}</p>
            <div className="modal-steps">
              <div>
                <Search width={24} height={24} />
                <span><strong>{t('how.step1Title')}</strong><small>{t('how.step1Text')}</small></span>
              </div>
              <div>
                <Database width={24} height={24} />
                <span><strong>{t('how.step2Title')}</strong><small>{t('how.step2Text')}</small></span>
              </div>
              <div>
                <ShieldCheck width={24} height={24} />
                <span><strong>{t('how.step3Title')}</strong><small>{t('how.step3Text')}</small></span>
              </div>
            </div>
            <button type="button" className="modal-action" onClick={() => setHowOpen(false)}>
              {t('how.confirm')} <ArrowRight width={17} height={17} />
            </button>
            <div className="modal-disclaimer">
              <CheckCircle width={15} height={15} /> {t('how.disclaimer')}
            </div>
          </section>
        </div>
      )}
    </div>
  );
}
