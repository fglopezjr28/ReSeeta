<?php

namespace App\Http\Requests;

use Illuminate\Foundation\Http\FormRequest;

class OCRImageRequest extends FormRequest
{
    public function authorize(): bool { return true; }

    public function rules(): array
    {
        return [
            'image' => [
                'required',
                'file',
                'mimes:png,jpg,jpeg,webp,bmp,tiff',
                'max:8192', // 8 MB
            ],
        ];
    }
}
