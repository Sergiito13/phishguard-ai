/**
 * PhishGuard AI - Internationalization
 *
 * Automatic browser language detection.
 * Persistent language preferences.
 */

import i18n from 'i18next';

import { initReactI18next } from 'react-i18next';

import LanguageDetector from 'i18next-browser-languagedetector';

import es from './locales/es.json';
import en from './locales/en.json';

import {
  DEFAULT_LANGUAGE,
  SUPPORTED_LANGUAGES,
} from './languages';


// ============================================================
// TRANSLATION RESOURCES
// ============================================================

const resources = {
  es: {
    translation: es,
  },
  en: {
    translation: en,
  },
};


// ============================================================
// UPDATE DOCUMENT LANGUAGE
// ============================================================

i18n.on('languageChanged', (language) => {
  document.documentElement.lang = language.split('-')[0];
});


// ============================================================
// INTERNATIONALIZATION CONFIGURATION
// ============================================================

void i18n
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    resources,

    supportedLngs: SUPPORTED_LANGUAGES.map(
      (language) => language.code
    ),

    fallbackLng: DEFAULT_LANGUAGE,

    load: 'languageOnly',

    detection: {
      order: [
        'localStorage',
        'navigator',
      ],

      lookupLocalStorage: 'phishguard-language',

      caches: [
        'localStorage',
      ],
    },

    interpolation: {
      escapeValue: false,
    },

    react: {
      useSuspense: false,
    },
  });

export default i18n;