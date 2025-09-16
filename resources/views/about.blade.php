<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>About - ReSeeta</title>
  <link rel="stylesheet" href="{{ asset('css/about.css') }}">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap" rel="stylesheet">
</head>
<body>
  <!-- Floating nav like landing -->
  <nav class="top-nav">
    <a href="{{ url('/') }}" class="{{ Request::is('/') ? 'active' : '' }}">Home</a>
    <a href="{{ url('/about') }}" class="{{ Request::is('about') ? 'active' : '' }}">About</a>
  </nav>

  <main class="about-section">
    <h1>Introducing <span>ReSeeta</span></h1>
    <p><strong>ReSeeta</strong> is a system that converts handwritten cursive medical prescriptions into 
      accurate digital text using a deep learning 
      <strong>ViT-CRNN (Vision Transformer-Convolutional Recurrent Neural Network) architecture.</strong> 
      Designed to convert even the most challenging cursive handwriting, it eliminates misinterpretations, 
      reduces errors, and streamlines healthcare workflows—because every life matters.</p>

    <p>By bridging the gap between paper and digital records, ReSeeta enhances patient safety, saves critical time 
      for medical professionals, and paves the way for smarter prescription management. Saving lives, one legible 
      prescription at a time.</p>
  </main>

  <!-- Decorative wave at bottom -->
  <svg class="bottom-wave"
       xmlns="http://www.w3.org/2000/svg"
       viewBox="0 0 1440 180"
       preserveAspectRatio="none">
    <path fill="#BCD6D0"
          d="M0,80
             C220,130 460,20 720,70
             S1220,140 1440,90
             L1440,180 L0,180 Z"/>
  </svg>

  <!-- Footer now last, below wave -->
  <footer>
    <p>© {{ date('Y') }} ReSeeta. All Rights Reserved.</p>
  </footer>
</body>
</html>
