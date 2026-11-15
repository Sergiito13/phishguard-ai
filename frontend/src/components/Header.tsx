import {
  OpenNewWindow,
  HalfMoon,
  SunLight,
} from 'iconoir-react';

import { useTranslation } from 'react-i18next';

import Brand from './Brand';
import LanguageSwitcher from './LanguageSwitcher';

interface HeaderProps {
  theme: 'dark' | 'light';
  onThemeToggle: () => void;
  onHome: () => void;
  onHow: () => void;
}

export default function Header({
  theme,
  onThemeToggle,
  onHome,
  onHow,
}: HeaderProps) {
  const { t } = useTranslation();

  const themeLabel = t(
    theme === 'dark'
      ? 'common.lightMode'
      : 'common.darkMode'
  );

  return (
    <header className="site-header">

      {/* LOGO */}

      <button
        className="brand-button"
        type="button"
        onClick={onHome}
        aria-label={t('common.home')}
      >
        <Brand theme={theme} />
      </button>

      {/* NAVIGATION */}

      <nav
        className="site-nav"
        aria-label="Main navigation"
      >
        <button
          type="button"
          className="site-nav__link"
          onClick={onHow}
        >
          {t('navigation.howItWorks')}
        </button>

        <a
          className="site-nav__link"
          href="https://github.com/Sergiito13/phishguard-ai#readme"
          target="_blank"
          rel="noopener noreferrer"
        >
          {t('navigation.documentation')}
        </a>

        <a
          className="site-nav__link site-nav__github"
          href="https://github.com/Sergiito13/phishguard-ai"
          target="_blank"
          rel="noopener noreferrer"
        >
          GitHub

          <OpenNewWindow
            width={13}
            height={13}
            strokeWidth={1.8}
          />
        </a>
      </nav>

      {/* LANGUAGE SWITCHER */}

      <LanguageSwitcher />

      {/* THEME SWITCHER */}

      <button
        className="theme-button"
        type="button"
        onClick={onThemeToggle}
        aria-label={themeLabel}
        title={themeLabel}
      >
        {theme === 'dark' ? (
          <HalfMoon
            width={20}
            height={20}
            strokeWidth={1.8}
          />
        ) : (
          <SunLight
            width={20}
            height={20}
            strokeWidth={1.8}
          />
        )}
      </button>

    </header>
  );
}