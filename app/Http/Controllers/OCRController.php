<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Http;
use Illuminate\Validation\Rule;

class OcrController extends Controller
{
    public function predict(Request $request)
    {
        $request->validate([
            'file'        => ['required','file','image','mimes:png,jpg,jpeg','max:10240'],
            'model'       => ['nullable', Rule::in(['vit','crnn'])],
            'use_context' => ['nullable','in:0,1'], // checkbox posts "1" or "0"
        ]);

        $file   = $request->file('file');
        $model  = $request->input('model', 'vit');

        // Context is only meaningful for ViT
        $useContext = $model === 'vit'
            ? ($request->input('use_context', '0') === '1' ? '1' : '0')
            : '0';

        // Pick the Python service
        $target = $model === 'crnn'
            ? rtrim(env('PYTHON_CRNN_URL', 'http://127.0.0.1:8002'), '/').'/predict'
            : rtrim(env('PYTHON_VIT_URL',  'http://127.0.0.1:8001'), '/').'/predict';

        // Send multipart: file + fields
        $response = Http::timeout(60)
            ->asMultipart() // IMPORTANT when mixing file + fields
            ->attach(
                'file',
                file_get_contents($file->getRealPath()),
                $file->getClientOriginalName()
            )
            ->post($target, [
                'model'        => $model,      // FastAPI expects 'model'
                'use_context'  => $useContext, // FastAPI expects 'use_context'
            ]);

        if (!$response->ok()) {
            return response()->json([
                'error'   => 'OCR service error',
                'details' => $response->body(),
            ], 502);
        }

        return response()->json($response->json());
    }
}
