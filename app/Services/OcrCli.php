<?php

namespace App\Services;

use Symfony\Component\Process\Process;
use Symfony\Component\Process\Exception\ProcessFailedException;

class OcrCli
{
    public function predict(string $imagePath): array
    {
        $python  = config('ocr.python');
        $script  = base_path(config('ocr.script'));   // ensure absolute
        $model   = base_path(config('ocr.model'));
        $weights = base_path(config('ocr.weights'));

        $cmd = [
            $python,
            $script,
            '--image', $imagePath,
            '--model', $model,
            '--weights', $weights,
            '--json'
        ];

        $process = new Process($cmd, base_path());
        $process->setTimeout(60);
        $process->run();

        if (!$process->isSuccessful()) {
            throw new ProcessFailedException($process);
        }

        $json = trim($process->getOutput());
        $data = json_decode($json, true);
        if (!is_array($data)) {
            throw new \RuntimeException("Invalid OCR JSON: ".$json);
        }
        return $data;
    }
}
