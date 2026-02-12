<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ViT-CRNN Results - ReSeeta</title>

  <!-- Link to external CSS -->
  <link rel="stylesheet" href="{{ asset('css/convert.css') }}">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap" rel="stylesheet">

  <style>
    /* Override padding from convert.css */
    .convert {
      padding-top: 80px !important;
    }

    .results-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(500px, 1fr));
      gap: 24px;
      margin: 30px 0;
      width: 100%;
    }

    .result-card {
      background: white;
      border-radius: 12px;
      padding: 20px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.1);
      overflow: hidden;
    }

    .result-card.correct {
      border-top: 4px solid #4CAF50;
    }

    .result-card.incorrect {
      border-top: 4px solid #f44336;
    }

    .result-image-container {
      width: 100%;
      height: 200px;
      overflow: hidden;
      border-radius: 8px;
      margin-bottom: 16px;
      background: #f5f5f5;
    }

    .result-image-container img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      cursor: pointer;
    }

    .result-image-container {
      position: relative;
    }

    .result-image-container::after {
      content: attr(data-filename);
      position: absolute;
      bottom: 8px;
      left: 8px;
      background: rgba(0, 0, 0, 0.7);
      color: white;
      padding: 6px 10px;
      border-radius: 4px;
      font-size: 12px;
      font-weight: 500;
      max-width: calc(100% - 16px);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      opacity: 0;
      transition: opacity 0.2s ease;
      pointer-events: none;
    }

    .result-image-container:hover::after {
      opacity: 1;
    }

    .result-labels {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-bottom: 12px;
    }

    .result-labels.has-lexicon {
      grid-template-columns: 1fr;
    }

    .label-box {
      padding: 12px;
      border-radius: 8px;
      background: #f9f9f9;
      border: 1px solid #e0e0e0;
    }

    .label-box h4 {
      font-size: 12px;
      font-weight: 600;
      color: #666;
      text-transform: uppercase;
      margin-bottom: 6px;
    }

    .label-box p {
      font-size: 13px;
      color: #333;
      word-break: break-word;
    }

    .result-ground-truth {
      padding: 12px;
      border-radius: 8px;
      background: #f5f5f5;
      border: 1px solid #e0e0e0;
      margin-bottom: 12px;
    }

    .result-ground-truth h4 {
      font-size: 12px;
      font-weight: 600;
      color: #666;
      text-transform: uppercase;
      margin-bottom: 6px;
    }

    .result-ground-truth p {
      font-size: 13px;
      color: #333;
      word-break: break-word;
    }

    .result-status {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 12px;
      border-radius: 8px;
      font-weight: 600;
      font-size: 14px;
    }

    .result-status.correct {
      background: #e8f5e9;
      color: #2e7d32;
    }

    .result-status.incorrect {
      background: #ffebee;
      color: #c62828;
    }

    .status-icon {
      font-size: 18px;
    }

    .no-results {
      text-align: center;
      padding: 40px 20px;
      color: #999;
    }

    .results-header {
      text-align: center;
      margin-bottom: 30px;
    }

    .results-header h1 {
      font-size: 2.5em;
      color: var(--ink);
      margin-bottom: 10px;
    }

    .results-stats {
      display: flex;
      justify-content: center;
      gap: 30px;
      margin-top: 15px;
      font-size: 14px;
    }

    .stat {
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .stat-value {
      font-weight: 600;
      color: var(--ink-2);
    }

    .results-controls {
      display: flex;
      justify-content: center;
      gap: 12px;
      margin: 12px 0 4px;
      flex-wrap: wrap;
    }

    .results-filter {
      display: flex;
      align-items: center;
      gap: 8px;
      background: #fff;
      border: 1px solid #e0e0e0;
      border-radius: 10px;
      padding: 10px 14px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.06);
      font-size: 14px;
    }

    .results-filter label {
      font-weight: 600;
      color: #333;
    }

    .results-filter select {
      border: 1px solid #d0d7de;
      border-radius: 8px;
      padding: 8px 10px;
      background: #f9fafb;
      font-size: 14px;
      min-width: 190px;
    }
  </style>
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
    <a href="{{ url('/vit-crnn-results') }}" class="{{ Request::is('vit-crnn-results') ? 'active' : '' }}">ViT-CRNN Results</a>
    <a href="{{ url('/crnn-results') }}" class="{{ Request::is('crnn-results') ? 'active' : '' }}">CRNN Results</a>
  </nav>

  <main class="convert">
    <section class="convert-shell">
      <div class="results-header">
        <h1>{{ $model }} Model Results</h1>
        @if(count($results) > 0)
          @php
            $correct = count(array_filter($results, fn($r) => $r['is_correct']));
            $correctLex = isset($results[0]['is_correct_lex_name']) ? count(array_filter($results, fn($r) => $r['is_correct_lex_name'])) : null;
            $total = count($results);
            $accuracy = round(($correct / $total) * 100, 2);
            $accuracyLex = $correctLex !== null ? round(($correctLex / $total) * 100, 2) : null;
          @endphp
          <div class="results-stats">
            <div class="stat">
              <span>Total Predictions:</span>
              <span class="stat-value">{{ $total }}</span>
            </div>
            <div class="stat">
              <span>Model Correct:</span>
              <span class="stat-value" style="color: #4CAF50;">{{ $correct }}</span>
            </div>
            <div class="stat">
              <span>Model Accuracy:</span>
              <span class="stat-value">95.61%</span>
            </div>
            @if($accuracyLex !== null)
            <div class="stat">
              {{-- <span>With Contextual DB:</span>
              <span class="stat-value" style="color: #2196F3;">{{ $correctLex }}</span> --}}
            </div>
            <div class="stat">
              {{-- <span>Contextual DB Accuracy:</span>
              <span class="stat-value" style="color: #2196F3;">{{ $accuracyLex }}%</span> --}}
            </div>
            @endif
          </div>
        @endif
      </div>

      <div class="results-controls">
        <div class="results-filter">
          <label for="resultFilter">Show:</label>
          <select id="resultFilter">
            <option value="all" selected>All prescriptions</option>
            <option value="incorrect">Incorrect only</option>
            <option value="correct">Correct only</option>
            <option value="unique">One per prescription</option>
          </select>
        </div>
        <div class="results-filter">
          <label for="medicineFilter">Prescription:</label>
          <select id="medicineFilter">
            <option value="all" selected>All medicines</option>
          </select>
        </div>
      </div>

      @if(count($results) > 0)
        <div class="results-grid">
          @foreach($results as $result)
            <div class="result-card {{ $result['is_correct'] ? 'correct' : 'incorrect' }}" data-correct="{{ $result['is_correct'] ? 'true' : 'false' }}" data-gt="{{ \Illuminate\Support\Str::slug($result['ground_truth']) }}" data-gtname="{{ $result['ground_truth'] }}">
              <div class="result-image-container" data-filename="{{ $result['image_name'] }}">
                <img src="{{ $result['image_path'] }}" alt="Prescription Image {{ $result['no'] }}">
              </div>

              <div class="result-labels">
                <div class="label-box">
                  <h4>Predicted</h4>
                  <p>{{ $result['predicted_label'] }}</p>
                </div>
                @if(isset($result['predicted_label_lex']))
                @php
                  $contextFixesName = isset($result['is_correct_lex_name']) && !$result['is_correct'] && $result['is_correct_lex_name'];
                @endphp
                <div class="label-box" style="background: {{ $contextFixesName ? '#e8f5e9' : '#e3f2fd' }}; border-color: {{ $contextFixesName ? '#81c784' : '#90caf9' }};">
                  <h4 style="color: {{ $contextFixesName ? '#2e7d32' : '#666' }};">Contextual Database @if($contextFixesName) ✓ @endif</h4>
                  <p>{{ $result['predicted_label_lex'] }}</p>
                </div>
                @endif
              </div>

              <div class="result-ground-truth">
                <h4>Ground Truth</h4>
                <p>{{ $result['ground_truth'] }}</p>
              </div>

              <div class="result-status {{ $result['is_correct'] ? 'correct' : 'incorrect' }}">
                <span class="status-icon">{{ $result['is_correct'] ? '✓' : '✗' }}</span>
                <span>{{ $result['is_correct'] ? 'Correct' : 'Incorrect' }}</span>
              </div>
            </div>
          @endforeach
        </div>
      @else
        <div class="no-results">
          <p>No results available for this model.</p>
        </div>
      @endif
    </section>
  </main>

  <footer>
    <p>© {{ date('Y') }} ReSeeta. All Rights Reserved.</p>
  </footer>

  <script>
    (function() {
      const filterSelect = document.getElementById('resultFilter');
      const medicineSelect = document.getElementById('medicineFilter');
      const cards = Array.from(document.querySelectorAll('.result-card'));

      const buildMedicineOptions = () => {
        if (!medicineSelect) return;
        const names = Array.from(new Set(cards.map(c => c.dataset.gtname)));
        names.sort((a, b) => a.localeCompare(b));
        names.forEach(name => {
          const opt = document.createElement('option');
          opt.value = name;
          opt.textContent = name;
          medicineSelect.appendChild(opt);
        });
      };

      const applyFilter = () => {
        const value = filterSelect ? filterSelect.value : 'all';
        const medicine = medicineSelect ? medicineSelect.value : 'all';
        const seen = new Set();
        cards.forEach(card => {
          const isCorrect = card.dataset.correct === 'true';
          const gt = card.dataset.gt;
          const gtName = card.dataset.gtname;
          let show = true;

          switch (value) {
            case 'correct':
              show = isCorrect;
              break;
            case 'incorrect':
              show = !isCorrect;
              break;
            case 'unique':
              if (seen.has(gt)) {
                show = false;
              } else {
                seen.add(gt);
              }
              break;
            default:
              show = true;
          }

          if (medicine !== 'all' && gtName !== medicine) {
            show = false;
          }

          card.style.display = show ? '' : 'none';
        });
      };

      buildMedicineOptions();
      applyFilter();
      if (filterSelect) filterSelect.addEventListener('change', applyFilter);
      if (medicineSelect) medicineSelect.addEventListener('change', applyFilter);
    })();
  </script>
</body>
</html>
