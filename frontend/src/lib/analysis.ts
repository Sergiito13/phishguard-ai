export type Classification = 'potential_phishing' | 'not_flagged' | 'not_analyzed';
export type ResultOrigin = 'sample' | 'preview' | 'api';

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

export const SAMPLE_URL = 'https://login-paypal-security.example.com/verify/account';
export const apiEnabled = import.meta.env.VITE_ANALYSIS_MODE === 'api';

/** A purely illustrative sample, not a run of the real ML classifier. */
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
      'Sample: familiar brand name embedded in a longer domain',
      'Sample: domain structure resembles an account verification page',
      'Sample: terms such as “login”, “security”, and “verify”',
      'Sample: path requests account verification',
    ],
    contributions: [
      { label: 'Brand-like domain pattern', value: 38 },
      { label: 'Account-related keywords', value: 24 },
      { label: 'URL length', value: 14 },
      { label: 'Hostname structure', value: 12 },
      { label: 'Special characters', value: 8 },
    ],
  };
}

export function parseURL(value: string): URL {
  const normalized = value.trim();
  if (!normalized) {
    throw new Error('Enter a URL to continue.');
  }
  if (normalized.length > 2048) {
    throw new Error('The URL is too long (maximum 2,048 characters).');
  }

  let parsed: URL;
  try {
    parsed = new URL(normalized);
  } catch {
    throw new Error('Enter a complete URL, for example https://example.com.');
  }

  if (!['http:', 'https:'].includes(parsed.protocol) || !(parsed.hostname.includes('.') || parsed.hostname.startsWith('['))) {
    throw new Error('Only complete HTTP or HTTPS URLs with a hostname are supported.');
  }
  return parsed;
}

/** Locally rendered URL information; no request is made to the submitted website. */
export function getURLFacts(raw: string) {
  const url = parseURL(raw);
  return {
    hostname: url.hostname,
    protocol: url.protocol.slice(0, -1).toUpperCase(),
    length: raw.length,
    path: `${url.pathname}${url.search}`,
    hostnameLabels: url.hostname.split('.').length,
    usesHTTPS: url.protocol === 'https:',
  };
}

export function createPreviewResult(raw: string): AnalysisResult {
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

function readString(value: unknown): string | null {
  return typeof value === 'string' && value.trim() ? value.trim() : null;
}

/** Contract designed for the future POST /api/v1/analyze endpoint. */
export async function analyzeURL(raw: string): Promise<AnalysisResult> {
  parseURL(raw);

  if (!apiEnabled) {
    return createPreviewResult(raw);
  }

  const base = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 15_000);

  try {
    const response = await fetch(`${base}/api/v1/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url: raw.trim() }),
      signal: controller.signal,
    });

    if (!response.ok) {
      throw new Error(`The analysis service returned HTTP ${response.status}.`);
    }

    const data: unknown = await response.json();
    if (typeof data !== 'object' || data === null) {
      throw new Error('The analysis service returned an invalid response.');
    }

    const payload = data as Record<string, unknown>;
    const label = payload.prediction ?? payload.classification;
    if (label !== 'potential_phishing' && label !== 'not_flagged') {
      throw new Error('Unknown classification returned by the analysis service.');
    }

    const score = payload.phishing_score;
    if (typeof score !== 'number' || !Number.isFinite(score) || score < 0 || score > 1) {
      throw new Error('Invalid model score returned by the analysis service.');
    }

    const signals = Array.isArray(payload.key_signals)
      ? payload.key_signals.filter((item): item is string => typeof item === 'string').slice(0, 5)
      : [];
    const impacts = Array.isArray(payload.feature_contributions)
      ? payload.feature_contributions.flatMap((item): FeatureImpact[] => {
          if (typeof item !== 'object' || item === null) return [];
          const contribution = item as Record<string, unknown>;
          return typeof contribution.label === 'string' &&
            typeof contribution.value === 'number' &&
            contribution.value >= 0 && contribution.value <= 100
            ? [{ label: contribution.label, value: contribution.value }]
            : [];
        }).slice(0, 6)
      : [];

    return {
      url: readString(payload.url) ?? raw.trim(),
      classification: label,
      phishingScore: score,
      threshold: typeof payload.threshold === 'number' ? payload.threshold : 0.5,
      calibrated: payload.calibrated === true,
      modelVersion: readString(payload.model_version) ?? 'unknown',
      origin: 'api',
      analyzedAt: new Date(),
      signals,
      contributions: impacts,
    };
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') {
      throw new Error(
        'The analysis service did not respond in time.',
        { cause: error }
      );
    }

    if (error instanceof TypeError) {
      throw new Error(
        'Could not connect to the analysis service. Check that FastAPI is running.',
        { cause: error }
      );
    }

    throw error;
  } finally {
    window.clearTimeout(timeout);
  }
}
