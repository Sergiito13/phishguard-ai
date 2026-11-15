/**
 * PhishGuard AI - URL Analysis
 *
 * URL validation, API integration, and analysis results.
 * Supports internationalized error messages.
 */

export type Classification =
  | 'potential_phishing'
  | 'not_flagged'
  | 'not_analyzed';

export type ResultOrigin = 'sample' | 'preview' | 'api';


// ============================================================
// TYPES
// ============================================================

export interface FeatureImpact {
  label: string;
  value: number;
}

export interface AnalysisResult {
  url: string;
  classification: Classification;
  phishingScore: number | null;
  threshold: number | null;
  calibrated: boolean;
  modelVersion: string;
  origin: ResultOrigin;
  analyzedAt: Date;
  signals: string[];
  contributions: FeatureImpact[];
}

export interface AnalysisFailure {
  key: string;
  options?: Record<string, string | number>;
}


// ============================================================
// TRANSLATABLE ANALYSIS ERRORS
// ============================================================

/**
 * Custom error for internationalized messages.
 *
 * Stores translation keys instead of hardcoded messages,
 * allowing errors to be displayed in the selected language.
 */
export class AnalysisError extends Error {
  readonly key: string;
  readonly options?: Record<string, string | number>;

  constructor(
    key: string,
    options?: Record<string, string | number>,
  ) {
    super(key);

    this.name = 'AnalysisError';
    this.key = key;
    this.options = options;
  }
}


// ============================================================
// CONFIGURATION
// ============================================================

export const SAMPLE_URL =
  'https://login-paypal-security.example.com/verify/account';

export const apiEnabled =
  import.meta.env.VITE_ANALYSIS_MODE === 'api';


// ============================================================
// SAMPLE ANALYSIS
// ============================================================

/**
 * Illustrative data only.
 *
 * Labels are translation keys and are translated
 * when the report is rendered.
 *
 * This is not a real Machine Learning prediction.
 */
export function createSampleResult(): AnalysisResult {
  return {
    url: SAMPLE_URL,
    classification: 'potential_phishing',
    phishingScore: 0.94,
    threshold: 0.5,
    calibrated: false,
    modelVersion: 'demo-only',
    origin: 'sample',
    analyzedAt: new Date(),

    signals: [
      'sample.signals.brand',
      'sample.signals.structure',
      'sample.signals.terms',
      'sample.signals.path',
    ],

    contributions: [
      {
        label: 'sample.contributions.brand',
        value: 38,
      },
      {
        label: 'sample.contributions.keywords',
        value: 24,
      },
      {
        label: 'sample.contributions.length',
        value: 14,
      },
      {
        label: 'sample.contributions.hostname',
        value: 12,
      },
      {
        label: 'sample.contributions.special',
        value: 8,
      },
    ],
  };
}


// ============================================================
// URL VALIDATION
// ============================================================

export function parseURL(value: string): URL {
  const normalized = value.trim();

  if (!normalized) {
    throw new AnalysisError('errors.emptyUrl');
  }

  if (normalized.length > 2048) {
    throw new AnalysisError('errors.tooLong');
  }

  let parsed: URL;

  try {
    parsed = new URL(normalized);
  } catch {
    throw new AnalysisError('errors.invalidUrl');
  }

  if (
    !['http:', 'https:'].includes(parsed.protocol) ||
    !(
      parsed.hostname.includes('.') ||
      parsed.hostname.startsWith('[')
    )
  ) {
    throw new AnalysisError('errors.unsupportedUrl');
  }

  return parsed;
}


// ============================================================
// URL FACTS
// ============================================================

/**
 * Extract locally rendered URL information.
 *
 * No HTTP requests are made to the destination website.
 */
export function getURLFacts(raw: string) {
  const url = parseURL(raw);

  return {
    hostname: url.hostname,

    protocol: url.protocol
      .slice(0, -1)
      .toUpperCase(),

    length: raw.length,

    path: `${url.pathname}${url.search}`,

    hostnameLabels: url.hostname.split('.').length,

    usesHTTPS: url.protocol === 'https:',
  };
}


// ============================================================
// PREVIEW ANALYSIS
// ============================================================

export function createPreviewResult(
  raw: string,
): AnalysisResult {
  parseURL(raw);

  return {
    url: raw.trim(),
    classification: 'not_analyzed',
    phishingScore: null,
    threshold: null,
    calibrated: false,
    modelVersion: 'not-connected',
    origin: 'preview',
    analyzedAt: new Date(),
    signals: [],
    contributions: [],
  };
}


// ============================================================
// HELPERS
// ============================================================

function readString(value: unknown): string | null {
  return typeof value === 'string' && value.trim()
    ? value.trim()
    : null;
}


// ============================================================
// MACHINE LEARNING API INTEGRATION
// ============================================================

/**
 * Analyze a URL using the FastAPI backend.
 *
 * Preserves the actual Machine Learning prediction.
 *
 * No fabricated signals or feature contributions
 * are generated by the frontend.
 */
export async function analyzeURL(
  raw: string,
): Promise<AnalysisResult> {
  // ----------------------------------------------------------
  // VALIDATE URL
  // ----------------------------------------------------------

  parseURL(raw);

  // ----------------------------------------------------------
  // PREVIEW MODE
  // ----------------------------------------------------------

  if (!apiEnabled) {
    return createPreviewResult(raw);
  }

  // ----------------------------------------------------------
  // API CONFIGURATION
  // ----------------------------------------------------------

  const base = (
    import.meta.env.VITE_API_BASE_URL ||
    'http://127.0.0.1:8000'
  ).replace(/\/$/, '');

  const controller = new AbortController();

  const timeout = window.setTimeout(
    () => controller.abort(),
    15_000,
  );

  try {
    // --------------------------------------------------------
    // SEND ANALYSIS REQUEST
    // --------------------------------------------------------

    const response = await fetch(
      `${base}/api/v1/analyze`,
      {
        method: 'POST',

        headers: {
          'Content-Type': 'application/json',
        },

        body: JSON.stringify({
          url: raw.trim(),
        }),

        signal: controller.signal,
      },
    );

    // --------------------------------------------------------
    // HANDLE HTTP ERRORS
    // --------------------------------------------------------

    if (!response.ok) {
      throw new AnalysisError(
        'errors.http',
        {
          status: response.status,
        },
      );
    }

    // --------------------------------------------------------
    // PARSE RESPONSE
    // --------------------------------------------------------

    const data: unknown = await response.json();

    if (
      typeof data !== 'object' ||
      data === null
    ) {
      throw new AnalysisError(
        'errors.invalidResponse',
      );
    }

    const payload = data as Record<string, unknown>;

    // --------------------------------------------------------
    // VALIDATE CLASSIFICATION
    // --------------------------------------------------------

    const label =
      payload.prediction ??
      payload.classification;

    if (
      label !== 'potential_phishing' &&
      label !== 'not_flagged'
    ) {
      throw new AnalysisError(
        'errors.unknownClassification',
      );
    }

    // --------------------------------------------------------
    // VALIDATE MODEL SCORE
    // --------------------------------------------------------

    const score = payload.phishing_score;

    if (
      typeof score !== 'number' ||
      !Number.isFinite(score) ||
      score < 0 ||
      score > 1
    ) {
      throw new AnalysisError(
        'errors.invalidScore',
      );
    }

    // --------------------------------------------------------
    // EXTRACT OPTIONAL SIGNALS
    // --------------------------------------------------------

    const signals = Array.isArray(
      payload.key_signals,
    )
      ? payload.key_signals
          .filter(
            (item): item is string =>
              typeof item === 'string',
          )
          .slice(0, 5)
      : [];

    // --------------------------------------------------------
    // EXTRACT OPTIONAL FEATURE CONTRIBUTIONS
    // --------------------------------------------------------

    const impacts = Array.isArray(
      payload.feature_contributions,
    )
      ? payload.feature_contributions
          .flatMap(
            (item): FeatureImpact[] => {
              if (
                typeof item !== 'object' ||
                item === null
              ) {
                return [];
              }

              const contribution =
                item as Record<string, unknown>;

              if (
                typeof contribution.label === 'string' &&
                typeof contribution.value === 'number' &&
                Number.isFinite(contribution.value) &&
                contribution.value >= 0 &&
                contribution.value <= 100
              ) {
                return [
                  {
                    label: contribution.label,
                    value: contribution.value,
                  },
                ];
              }

              return [];
            },
          )
          .slice(0, 6)
      : [];

    // --------------------------------------------------------
    // BUILD ANALYSIS RESULT
    // --------------------------------------------------------

    return {
      url: readString(payload.url) ?? raw.trim(),

      classification: label,

      phishingScore: score,

      threshold:
        typeof payload.threshold === 'number'
          ? payload.threshold
          : 0.5,

      calibrated: payload.calibrated === true,

      modelVersion:
        readString(payload.model_version) ??
        'unknown',

      origin: 'api',

      analyzedAt: new Date(),

      signals,

      contributions: impacts,
    };
  } catch (error) {
    // --------------------------------------------------------
    // REQUEST TIMEOUT
    // --------------------------------------------------------

    if (
      error instanceof Error &&
      error.name === 'AbortError'
    ) {
      throw new AnalysisError(
        'errors.timeout',
      );
    }

    // --------------------------------------------------------
    // CONNECTION ERROR
    // --------------------------------------------------------

    if (error instanceof TypeError) {
      throw new AnalysisError(
        'errors.connection',
      );
    }

    // --------------------------------------------------------
    // OTHER ERRORS
    // --------------------------------------------------------

    throw error;
  } finally {
    // --------------------------------------------------------
    // CLEANUP
    // --------------------------------------------------------

    window.clearTimeout(timeout);
  }
}