/**
 * PhishGuard AI - Supported Languages
 */

import spanishFlag from 'flag-icons/flags/4x3/es.svg?url';
import britishFlag from 'flag-icons/flags/4x3/gb.svg?url';

export const SUPPORTED_LANGUAGES = [
  {
    code: 'es',
    name: 'Español',
    shortName: 'ES',
    flag: spanishFlag,
  },
  {
    code: 'en',
    name: 'English',
    shortName: 'EN',
    flag: britishFlag,
  },
] as const;

export type SupportedLanguage =
  (typeof SUPPORTED_LANGUAGES)[number]['code'];

export const DEFAULT_LANGUAGE: SupportedLanguage = 'en';