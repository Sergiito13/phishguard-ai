/**
 * PhishGuard AI - Home
 *
 * Main application screen.
 * Supports Spanish and English translations.
 */

import {
  ArrowRight,
  Flash,
  NavArrowRight,
  ViewStructureDown,
  Link,
  ShieldCheck,
  Sparks,
} from 'iconoir-react';

import {
  useState,
  type FormEvent,
} from 'react';

import { useTranslation } from 'react-i18next';

import {
  apiEnabled,
  type AnalysisFailure,
} from '../lib/analysis';


// ============================================================
// PROPS
// ============================================================

interface HomeProps {
  onAnalyze: (url: string) => Promise<void>;
  onSample: () => void;
  busy: boolean;
  error: AnalysisFailure | null;
}


// ============================================================
// FEATURES
// ============================================================

const FEATURES = [
  {
    icon: Flash,
    name: 'fast',
  },
  {
    icon: ViewStructureDown,
    name: 'ai',
  },
  {
    icon: ShieldCheck,
    name: 'privacy',
  },
] as const;


// ============================================================
// HOME COMPONENT
// ============================================================

export default function Home({
  onAnalyze,
  onSample,
  busy,
  error,
}: HomeProps) {
  const { t } = useTranslation();

  const [url, setUrl] = useState('');


  // ==========================================================
  // FORM SUBMISSION
  // ==========================================================

  function onSubmit(
    event: FormEvent<HTMLFormElement>,
  ) {
    event.preventDefault();

    void onAnalyze(url);
  }


  // ==========================================================
  // RENDER
  // ==========================================================

  return (
    <main className="home-screen">

      {/* ====================================================
          HERO SECTION
          ==================================================== */}

      <section
        className="hero"
        aria-labelledby="hero-title"
      >

        {/* EYEBROW */}

        <div className="eyebrow">
          <Sparks
            width={16}
            height={16}
            strokeWidth={1.8}
          />

          <span>
            {t('home.eyebrow')}
          </span>
        </div>


        {/* TITLE */}

        <h1 id="hero-title">
          {t('home.title.first')}

          <br />

          {t('home.title.second')}

          <span>
            {t('home.title.highlight')}
          </span>
        </h1>


        {/* DESCRIPTION */}

        <p className="hero__description">
          {t('home.description')}
        </p>


        {/* ==================================================
            URL ANALYSIS FORM
            ================================================== */}

        <form
          className="url-form"
          onSubmit={onSubmit}
        >
          <Link
            className="url-form__icon"
            width={27}
            height={27}
            strokeWidth={2.1}
            aria-hidden="true"
          />

          <label
            className="visually-hidden"
            htmlFor="target-url"
          >
            {t('home.urlLabel')}
          </label>

          <input
            id="target-url"
            type="url"
            placeholder={t('home.urlPlaceholder')}
            spellCheck={false}
            autoComplete="url"
            value={url}
            onChange={(event) => {
              setUrl(event.target.value);
            }}
            maxLength={2048}
            disabled={busy}
            required
          />

          <button
            type="submit"
            className="primary-button"
            disabled={busy}
          >
            <span>
              {busy
                ? t('home.analyzing')
                : t('home.analyze')}
            </span>

            {busy ? (
              <span
                className="button-spinner"
                aria-hidden="true"
              />
            ) : (
              <ArrowRight
                width={19}
                height={19}
                strokeWidth={1.9}
              />
            )}
          </button>
        </form>


        {/* ==================================================
            ERROR MESSAGE
            ================================================== */}

        {error && (
          <p
            className="form-error"
            role="alert"
          >
            {t(
              error.key,
              error.options ?? {},
            )}
          </p>
        )}


        {/* ==================================================
            API STATUS
            ================================================== */}

        <div className="hero__caption">

          <span className="mode-dot" />

          {apiEnabled
            ? t('home.connected')
            : t('home.notConnected')}

          <span className="hero__caption-divider">
            ·
          </span>

          <button
            type="button"
            onClick={onSample}
          >
            {t('home.sampleReport')}

            <NavArrowRight
              width={13}
              height={13}
            />
          </button>
        </div>

      </section>


      {/* ====================================================
          FEATURE CARDS
          ==================================================== */}

      <section
        className="feature-grid"
        aria-label={t('home.highlights')}
      >
        {FEATURES.map(
          ({ icon: Icon, name }) => (
            <article
              className="feature-card"
              key={name}
            >
              <Icon
                width={34}
                height={34}
                strokeWidth={1.7}
                className="feature-card__icon"
                aria-hidden="true"
              />

              <h2>
                {t(`home.features.${name}.title`)}
              </h2>

              <p>
                {t(`home.features.${name}.description`)}
              </p>
            </article>
          ),
        )}
      </section>


      {/* ====================================================
          FOOTER
          ==================================================== */}

      <p className="home-footer">
        {t('home.footer.open')}

        <span>·</span>

        {t('home.footer.developers')}

        <span>·</span>

        {t('home.footer.improving')}
      </p>

    </main>
  );
}