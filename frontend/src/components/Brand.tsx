import logoDark from '../assets/phishguard-logo-dark.svg';
import logoLight from '../assets/phishguard-logo-light.svg';

export default function Brand({ theme }: { theme: 'dark' | 'light' }) {
  return (
    <img
      className="brand__logo"
      src={theme === 'dark' ? logoDark : logoLight}
      alt="PhishGuard AI"
      width={195}
      height={50}
    />
  );
}
