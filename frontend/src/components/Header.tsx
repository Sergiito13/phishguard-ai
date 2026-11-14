import { OpenNewWindow, HalfMoon, SunLight } from 'iconoir-react';
import Brand from './Brand';

interface HeaderProps {
  theme: 'dark' | 'light';
  onThemeToggle: () => void;
  onHome: () => void;
  onHow: () => void;
}

export default function Header({ theme, onThemeToggle, onHome, onHow }: HeaderProps) {
  return (
    <header className="site-header">
      <button className="brand-button" type="button" onClick={onHome} aria-label="Go to PhishGuard AI home">
        <Brand theme={theme} />
      </button>
      <nav className="site-nav" aria-label="Main navigation">
        <button type="button" className="site-nav__link" onClick={onHow}>How it works</button>
        <a className="site-nav__link" href="https://github.com/Sergiito13/phishguard-ai#readme" target="_blank" rel="noopener noreferrer">Documentation</a>
        <a className="site-nav__link site-nav__github" href="https://github.com/Sergiito13/phishguard-ai" target="_blank" rel="noopener noreferrer">GitHub <OpenNewWindow width={13} height={13} strokeWidth={1.8} /></a>
      </nav>
      <button className="theme-button" type="button" onClick={onThemeToggle} aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`} title="Toggle theme">
        {theme === 'dark' ? <HalfMoon width={20} height={20} strokeWidth={1.8} /> : <SunLight width={20} height={20} strokeWidth={1.8} />}
      </button>
    </header>
  );
}
