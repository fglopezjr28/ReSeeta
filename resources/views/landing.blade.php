<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ReSeeta</title>

  <!-- Link to external CSS -->
  <link rel="stylesheet" href="{{ asset('css/landing.css') }}">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap" rel="stylesheet">
</head>
<body>
  <!-- Prototype decorative curve -->
  <svg class="bg-curve"
       xmlns="http://www.w3.org/2000/svg"
       viewBox="-120 -40 1400 950"
       preserveAspectRatio="xMinYMin slice"
       aria-hidden="true">
    <g transform="translate(-10,-10) scale(0.985)">
      <path d="M1099 397.584C277.924 461.68 2298.66 700.137 1155 876.084C635 956.084 61.1822 843.575 -137 597C-309 382.999 -475 -271 654 55.5001C1051.22 170.375 1375.5 376 1099 397.584Z"
            fill="#FAFFFE"/>
    </g>
  </svg>
  
  <!-- Floating nav -->
  <nav class="top-nav">
    <a href="{{ url('/') }}" class="{{ Request::is('/') ? 'active' : '' }}">Home</a>
    <a href="{{ url('/about') }}" class="{{ Request::is('about') ? 'active' : '' }}">About</a>
  </nav>

  <main class="hero">
    <div class="left-column">
      <img src="{{ asset('assets/eye.png') }}" alt="ReSeeta Logo" class="eye-logo">

      <!-- <h1 class="brand-title">ReSeeta</h1> -->
      <p class="tagline">
        <strong>“WE CONVERT</strong> because every life matters — Saving Lives, One 
        Legible Prescription at a Time.<strong>”</strong>
      </p>
    </div>

    <div class="right-column">
      <a href="{{ url('/convert') }}" class="convert-btn" title="Start Converting Prescriptions">Start Converting</a>
    </div>
  </main>

  <footer>
    <p>© {{ date('Y') }} ReSeeta. All Rights Reserved.</p>
  </footer>
</body>
</html>
