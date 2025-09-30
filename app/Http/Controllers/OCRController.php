<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Http;

class OcrController extends Controller
{
    public function predict(Request $request)
    {
        $request->validate([
            'file' => ['required','file','image','mimes:png,jpg,jpeg','max:10240'], // 10MB
        ]);

        $file = $request->file('file');

        // Send to Python FastAPI
        $response = Http::timeout(30)->attach(
            'file', file_get_contents($file->getRealPath()), $file->getClientOriginalName()
        )->post(env('PYTHON_OCR_URL', 'http://127.0.0.1:8001').'/predict');

        if (!$response->ok()) {
            return response()->json(['error' => 'OCR service error', 'details' => $response->body()], 502);
        }

        return response()->json($response->json());
    }
}
