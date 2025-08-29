<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ReSeeta</title>
  <link rel="stylesheet" href="{{ asset('css/convert.css') }}">
</head>
<body>
  <header>
    <a href="{{ url('/') }}">Home</a>
    <a href="{{ url('/about') }}">About</a>
  </header>

  <main>
    <div class="container">
      <div class="left-column upload-container">
        <h2>Upload Prescription</h2>
        <input type="file" id="fileInput" accept="image/*">
        <div class="image-preview">
          <img id="previewImage" src="" alt="Image Preview" style="display:none; max-width:100%; margin-top:10px; border-radius:8px;">
        </div>
      </div>

      <div class="right-column result-container">
        <h2>Result</h2>
        <div class="result-box">
          <!-- Recognized text will be displayed here -->
        </div>
      </div>
    </div>

    <div class="button-container">
      <button id="startConvert">Recognize Prescription</button>
    </div>
  </main>

  <footer>
    <p>© 2025 ReSeeta. All Rights Reserved.</p>
  </footer>

  <script>
    const fileInput = document.getElementById('fileInput');
    const previewImage = document.getElementById('previewImage');

    fileInput.addEventListener('change', function(event) {
      const file = event.target.files[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
          previewImage.src = e.target.result;
          previewImage.style.display = 'block';
        };
        reader.readAsDataURL(file);
      }
    });
  </script>
</body>
</html>
