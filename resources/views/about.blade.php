<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>About - ReSeeta</title>
  <link rel="stylesheet" href="{{ asset('css/about.css') }}">
</head>
<body>
  <header>
    <a href="{{ url('/') }}" class="{{ Request::is('/') ? 'active' : '' }}">Home</a>
    <a href="{{ url('/about') }}" class="{{ Request::is('about') ? 'active' : '' }}">About</a>
  </header>

  <main>
    <h1>Introducing ReSeeta</h1>
    <p>ReSeeta is a system that converts handwritten cursive medical prescriptions into accurate digital text 
      using a deep learning ViT-CRNN (Vision Transformer-Convolutional Recurrent Neural Network) architecture. 
      Designed to convert even the most challenging cursive handwriting, it eliminates misinterpretations, 
      reduces errors, and streamlines healthcare workflows—because every life matters. </p>
    <p>By bridging the gap between paper and digital records, ReSeeta enhances patient safety, saves critical time 
      for medical professionals, and paves the way for smarter prescription management. Saving lives, one legible 
      prescription at a time.</p>
  </main>

  <footer>
    <p>&copy; 2025 ReSeeta. All Rights Reserved.</p>
  </footer>
</body>
</html>
