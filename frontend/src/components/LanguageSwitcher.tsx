import { useEffect, useRef, useState } from 'react';
import { Check, NavArrowDown } from 'iconoir-react';
import { useTranslation } from 'react-i18next';

import { SUPPORTED_LANGUAGES } from '../i18n/languages';

export default function LanguageSwitcher() {
  const { t, i18n } = useTranslation();

  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const currentLanguage =
    i18n.resolvedLanguage?.split('-')[0] ?? 'en';

  const selectedLanguage =
    SUPPORTED_LANGUAGES.find(
      (language) => language.code === currentLanguage
    ) ?? SUPPORTED_LANGUAGES[1];

  useEffect(() => {
    if (!open) return;

    const handleOutsideClick = (event: PointerEvent) => {
      if (
        containerRef.current &&
        !containerRef.current.contains(event.target as Node)
      ) {
        setOpen(false);
      }
    };

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setOpen(false);
      }
    };

    document.addEventListener('pointerdown', handleOutsideClick);
    document.addEventListener('keydown', handleKeyDown);

    return () => {
      document.removeEventListener('pointerdown', handleOutsideClick);
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [open]);

  const changeLanguage = async (language: string) => {
    await i18n.changeLanguage(language);
    setOpen(false);
  };

  return (
    <div className="language-switcher" ref={containerRef}>
      <button
        type="button"
        className="language-button"
        aria-label={t('common.selectLanguage')}
        aria-expanded={open}
        aria-controls="language-menu"
        onClick={() => setOpen((current) => !current)}
      >
        <img
          src={selectedLanguage.flag}
          alt=""
          className="language-flag"
        />

        <span className="language-code">
          {selectedLanguage.shortName}
        </span>

        <NavArrowDown width={14} height={14} />
      </button>

      {open && (
        <div
          id="language-menu"
          className="language-menu"
          role="group"
          aria-label={t('common.selectLanguage')}
        >
          {SUPPORTED_LANGUAGES.map((language) => {
            const active = currentLanguage === language.code;

            return (
              <button
                key={language.code}
                type="button"
                lang={language.code}
                className={`language-option ${
                  active ? 'language-option--active' : ''
                }`}
                aria-pressed={active}
                onClick={() => void changeLanguage(language.code)}
              >
                <img
                  src={language.flag}
                  alt=""
                  className="language-flag"
                />

                <span>{language.name}</span>

                {active && <Check width={17} height={17} />}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}