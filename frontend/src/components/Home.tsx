import { ArrowRight, Flash, NavArrowRight, ViewStructureDown, Link, ShieldCheck, Sparks } from 'iconoir-react';
import { useState, type FormEvent } from 'react';
import { apiEnabled } from '../lib/analysis';

interface HomeProps {
  onAnalyze: (url: string) => Promise<void>;
  onSample: () => void;
  busy: boolean;
  error: string | null;
}

const FEATURES = [
  { icon: Flash, title: 'Fast & focused', description: 'Explore URL patterns in seconds.' },
  { icon: ViewStructureDown, title: 'AI-powered', description: 'Built around a machine learning model.' },
  { icon: ShieldCheck, title: 'Privacy first', description: 'We inspect the URL text, not the page.' },
];

export default function Home({ onAnalyze, onSample, busy, error }: HomeProps) {
  const [url, setUrl] = useState('');
  function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void onAnalyze(url);
  }

  return (
    <main className="home-screen">
      <section className="hero" aria-labelledby="hero-title">
        <div className="eyebrow"><Sparks width={16} height={16} strokeWidth={1.8} /><span>AI-POWERED SECURITY</span></div>
        <h1 id="hero-title">Think before<br />you <span>click.</span></h1>
        <p className="hero__description">Intelligent phishing detection. Analyze suspicious<br className="desktop-break" /> links in seconds with machine learning.</p>

        <form className="url-form" onSubmit={onSubmit}>
          <Link className="url-form__icon" width={27} height={27} strokeWidth={2.1} aria-hidden="true" />
          <label className="visually-hidden" htmlFor="target-url">URL to analyze</label>
          <input
            id="target-url"
            type="url"
            placeholder="Enter a URL (e.g. https://example.com)"
            spellCheck={false}
            autoComplete="url"
            value={url}
            onChange={(event) => setUrl(event.target.value)}
            maxLength={2048}
            disabled={busy}
            required
          />
          <button type="submit" className="primary-button" disabled={busy}>
            <span>{busy ? 'Analyzing…' : 'Analyze URL'}</span>
            {busy ? <span className="button-spinner" aria-hidden="true" /> : <ArrowRight width={19} height={19} strokeWidth={1.9} />}
          </button>
        </form>
        {error && <p className="form-error" role="alert">{error}</p>}
        <div className="hero__caption">
          <span className="mode-dot" />
          {apiEnabled ? 'Connected to the experimental model API' : 'Frontend preview · Model not connected yet'}
          <span className="hero__caption-divider">·</span>
          <button type="button" onClick={onSample}>View sample report <NavArrowRight width={13} height={13} /></button>
        </div>
      </section>

      <section className="feature-grid" aria-label="Highlights">
        {FEATURES.map((feature) => (
          <article className="feature-card" key={feature.title}>
            <feature.icon width={34} height={34} strokeWidth={1.7} className="feature-card__icon" aria-hidden="true" />
            <h2>{feature.title}</h2>
            <p>{feature.description}</p>
          </article>
        ))}
      </section>
      <p className="home-footer">Open source <span>·</span> Built for developers <span>·</span> Continuously improving</p>
    </main>
  );
}
