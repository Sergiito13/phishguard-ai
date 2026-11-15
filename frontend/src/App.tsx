import { useEffect, useState } from 'react';
import { Navigate, Route, Routes, useLocation, useNavigate } from 'react-router-dom';
import { ArrowRight, CheckCircle, Database, Search, ShieldCheck, Xmark } from 'iconoir-react';
import Header from './components/Header';
import BackgroundArt from './components/BackgroundArt';
import Home from './components/Home';
import Report from './components/Report';
import { analyzeURL, createPreviewResult, createSampleResult, type AnalysisResult } from './lib/analysis';

type Theme = 'dark' | 'light';
type RouteState = { result?: AnalysisResult } | null;

export default function App() {
  const [theme, setTheme] = useState<Theme>('dark');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
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
      navigate('/analysis', { state: { result } });
      window.scrollTo({ top: 0, behavior: 'smooth' });
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'An unexpected error occurred.');
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
  const queryURL = params.get('url');
  let result: AnalysisResult | null = rememberedResult ?? null;
  if (!result && params.get('sample') === '1') result = createSampleResult();
  if (!result && queryURL) {
    try { result = createPreviewResult(queryURL); } catch { result = null; }
  }

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
        <div className="modal-overlay" role="presentation" onMouseDown={event => { if (event.target === event.currentTarget) setHowOpen(false); }}>
          <section className="how-modal" role="dialog" aria-modal="true" aria-labelledby="how-title">
            <button type="button" className="modal-close" onClick={() => setHowOpen(false)} aria-label="Close dialog"><Xmark width={21} height={21} /></button>
            <span className="modal-kicker">UNDER THE HOOD</span>
            <h2 id="how-title">Simple input.<br /><span>Thoughtful analysis.</span></h2>
            <p className="modal-lead">PhishGuard AI is an experimental URL-classification project built around structural features and machine learning.</p>
            <div className="modal-steps">
              <div><Search width={24} height={24} /><span><strong>01 / Enter a link</strong><small>We validate and parse the URL, without browsing the destination.</small></span></div>
              <div><Database width={24} height={24} /><span><strong>02 / Extract patterns</strong><small>The Python backend extracts 20 lexical and structural features.</small></span></div>
              <div><ShieldCheck width={24} height={24} /><span><strong>03 / Review an experimental verdict</strong><small>The model can flag a URL, but cannot guarantee safety.</small></span></div>
            </div>
            <button type="button" className="modal-action" onClick={() => setHowOpen(false)}>Got it <ArrowRight width={17} height={17} /></button>
            <div className="modal-disclaimer"><CheckCircle width={15} height={15} /> Designed to integrate with FastAPI in the next phase.</div>
          </section>
        </div>
      )}
    </div>
  );
}
