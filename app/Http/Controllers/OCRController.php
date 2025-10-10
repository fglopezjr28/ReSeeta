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
            'file'  => ['required','file','image','mimes:png,jpg,jpeg','max:10240'], // 10 MB
            'model' => ['nullable', Rule::in(['vit','crnn'])],                      // defaults to 'vit' below
        ]);

        $file  = $request->file('file');
        $model = $request->input('model', 'vit'); // default ViT for backward-compat

        // Decide which Python service to call
        $target = $model === 'crnn'
            ? rtrim(env('PYTHON_CRNN_URL', 'http://127.0.0.1:8002'), '/').'/predict'
            : rtrim(env('PYTHON_VIT_URL',  'http://127.0.0.1:8001'), '/').'/predict';

        // Send multipart with the image file
        $response = Http::timeout(60)
            ->attach(
                'file',
                file_get_contents($file->getRealPath()),
                $file->getClientOriginalName()
            )
            ->post($target);

        if (!$response->ok()) {
            return response()->json([
                'error'   => 'OCR service error',
                'details' => $response->body(),
            ], 502);
        }

        return response()->json($response->json());
    }
}
