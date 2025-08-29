<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ReSeeta</title>

  <!-- Link to external CSS -->
  <link rel="stylesheet" href="{{ asset('css/landing.css') }}">
</head>
<body>
  <header>
    <a href="#" class="home-link active">Home</a>
    <a href="{{ url('/about') }}" class="{{ Request::is('about') ? 'active' : '' }}">About</a>
  </header>

  <main>
    <div class="left-column">
      <div class="logo">
        <img src="{{ asset('assets/eye.png') }}" alt="ReSeeta Logo" class="eye-logo">
      </div>

      <p class="tagline">
        <strong>“WE CONVERT</strong> because every life matters — Saving Lives, One 
        Legible Prescription at a Time.”
      </p>
    </div>

    <div class="right-column">
      <button id="startConvert" class="convert-btn">Start Converting</button>
    </div>
  </main>

  <footer>
    <p>© 2025 ReSeeta. All Rights Reserved.</p>
  </footer>

<script>
  document.getElementById('startConvert').addEventListener('click', function() {
    window.location.href = "{{ url('/convert') }}";
  });
</script>
</body>
</html>
