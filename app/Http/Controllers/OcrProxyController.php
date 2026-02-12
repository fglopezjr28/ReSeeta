<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Http;

class OcrProxyController extends Controller
{
    /**
     * POST /ocr/predict
     * Proxy upload to FastAPI and write the FastAPI JSON body to public/ocr_result.json
     *
     * Requirements:
     *  - FASTAPI_URL in .env (defaults to http://127.0.0.1:8001/predict_both)
     *  - public/ must be writable by PHP so the file can be created/overwritten
     */
    public function predict(Request $request)
    {
        if (!$request->hasFile('file')) {
            return response()->json(['error' => 'No file uploaded. Please attach a "file" field.'], 400);
        }

        $fastapi = env('FASTAPI_URL', 'http://127.0.0.1:8001/predict_both');

        // configure client; increase timeout for remote OCR polling
        $client = Http::timeout(180);

        try {
            $file = $request->file('file');
            $fh = fopen($file->getRealPath(), 'r');

            // Attach file and forward rest of form fields (except _token/_method)
            $payload = $this->forwardFormFields($request);

            $resp = $client->withHeaders([
                // add any headers if needed
            ])->attach(
                'file',
                $fh,
                $file->getClientOriginalName()
            )->post($fastapi, $payload);

            fclose($fh);

            $body = $resp->body();
            $status = $resp->status();

            // Attempt to save raw body to public/ocr_result.json for frontend fetching
            try {
                @file_put_contents(public_path('ocr_result.json'), $body);
            } catch (\Throwable $e) {
                // non-fatal — do not break the proxied response
                // \Log::warning("Failed to write ocr_result.json: " . $e->getMessage());
            }

            // Normalize response: if FastAPI returned JSON, forward as JSON with same status
            $decoded = json_decode($body, true);
            if (json_last_error() === JSON_ERROR_NONE) {
                return response()->json($decoded, $status);
            }

            // otherwise return raw body with content-type forwarded if present
            $contentType = $resp->header('Content-Type') ?: 'text/plain';
            return response($body, $status)->header('Content-Type', $contentType);

        } catch (\Throwable $e) {
            // Proxy error (e.g. connection, timeout). Return 502 with details.
            return response()->json([
                'ok' => false,
                'error' => 'Proxy request failed',
                'detail' => $e->getMessage()
            ], 502);
        }
    }

    /**
     * Extract form fields to forward (exclude file and Laravel-only fields)
     */
    protected function forwardFormFields(Request $request): array
    {
        $input = $request->except(['file', '_token', '_method']);
        array_walk($input, function (&$v) {
            if (is_array($v)) $v = implode(',', $v);
        });
        return $input;
    }
}
