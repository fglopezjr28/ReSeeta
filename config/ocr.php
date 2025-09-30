<?php

return [
    'python'  => env('OCR_PYTHON', base_path('.venv/bin/python')),
    'script'  => env('OCR_SCRIPT', base_path('py-model/predict.py')),
    'model'   => env('OCR_MODEL',  base_path('py-model/reseeta_model.py')),
    'weights' => env('OCR_WEIGHTS', base_path('py-model/weights/ViT_CRNN_weights.pth')),
];