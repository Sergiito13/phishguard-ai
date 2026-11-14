export default function BackgroundArt() {
  return (
    <div className="background-art" aria-hidden="true">
      <div className="background-art__glow background-art__glow--one" />
      <div className="background-art__glow background-art__glow--two" />
      <svg className="background-art__waves" viewBox="0 0 1400 960" preserveAspectRatio="xMidYMid slice">
        <defs>
          <linearGradient id="heroWave" x1=".2" y1=".9" x2=".9" y2=".1">
            <stop offset="0" stopColor="#053A3A" stopOpacity="0" />
            <stop offset=".63" stopColor="#087C79" stopOpacity=".38" />
            <stop offset="1" stopColor="#23C6BB" stopOpacity=".85" />
          </linearGradient>
          <linearGradient id="lowerWave" x1="0" y1="1" x2="1" y2="0">
            <stop offset="0" stopColor="#19E4CB" stopOpacity=".6" />
            <stop offset="1" stopColor="#03504F" stopOpacity="0" />
          </linearGradient>
        </defs>
        <path d="M1400 95 C1160 158 1114 260 978 335 C790 438 680 295 535 420 C330 597 107 547 -40 680 L-40 0 1400 0Z" fill="url(#heroWave)" opacity=".4" />
        <path d="M1406 203 C1140 214 1070 447 875 480 C730 505 587 384 453 487" stroke="#53DACE" strokeOpacity=".36" strokeWidth="1" fill="none" />
        <path d="M-15 772 C148 791 198 861 340 982 H-15Z" fill="url(#lowerWave)" />
        <path d="M-15 816 C154 824 240 928 356 990" stroke="#5AE4D0" strokeOpacity=".55" fill="none" strokeWidth="1" />
      </svg>
    </div>
  );
}
