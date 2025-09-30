<?php

namespace App\Services;

use Illuminate\Http\Client\PendingRequest;
use Illuminate\Http\UploadedFile;
use Illuminate\Support\Facades\Http;

class PyModelClient
{
    protected PendingRequest $http;
    protected string $base;

    public function __construct()
    {
        $cfg = config('services.py_model');
        $this->base = rtrim($cfg['base_url'], '/');

        $http = Http::timeout($cfg['timeout'] ?? 15)
            ->baseUrl($this->base)
            ->acceptJson();

        if (!empty($cfg['api_key'])) {
            $http = $http->withHeaders(['X-API-Key' => $cfg['api_key']]);
        }

        $this->http = $http;
    }

    public function health(): array
    {
        return $this->http->get('/health')->throw()->json();
    }

    public function predictImage(UploadedFile $image): array
    {
        // forwards to FastAPI /predict-image
        $resp = $this->http->asMultipart()
            ->attach(
                'file',
                file_get_contents($image->getRealPath()),
                $image->getClientOriginalName()
            )
            ->post('/predict-image');

        // throw on non-2xx to bubble up a proper error
        return $resp->throw()->json();
    }

    // Optional: base64 endpoint
    public function predictImageB64(string $dataUrlOrB64): array
    {
        return $this->http->post('/predict-image-b64', [
            'image_b64' => $dataUrlOrB64,
        ])->throw()->json();
    }
}
