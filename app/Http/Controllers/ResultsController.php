<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;

class ResultsController extends Controller
{
    public function vitCrnnResults()
    {
        $results = $this->parseResultsCSV('Combined_Test_Set_Recognition_Results_lexicon.csv', true);
        return view('vit-crnn-results', ['results' => $results, 'model' => 'ViT-CRNN']);
    }

    public function crnnResults()
    {
        $results = $this->parseResultsCSV('CRNN_Combined_Test_Set_Recognition_Results.csv', false);
        return view('crnn-results', ['results' => $results, 'model' => 'CRNN']);
    }

    private function parseResultsCSV($filename, $hasLexicon = false)
    {
        $filePath = public_path('results/' . $filename);
        
        if (!file_exists($filePath)) {
            return [];
        }

        $results = [];
        $file = fopen($filePath, 'r');
        $header = fgetcsv($file); // Skip header
        
        $currentLine = 2; // Start counting from line 2 (after header)

        while (($row = fgetcsv($file)) !== false) {
            // Include all results
            if (count($row) >= 4) {
                $imageName = trim($row[1]);
                $groundTruth = trim($row[2]);
                $predicted = trim($row[3]);
                $predictedLex = $hasLexicon && isset($row[4]) ? trim($row[4]) : null;

                // Full-string correctness
                $isCorrect = strcasecmp($groundTruth, $predicted) === 0;
                $isCorrectLex = $predictedLex ? strcasecmp($groundTruth, $predictedLex) === 0 : null;

                // Medicine-name-only correctness (before the first ':')
                $gtName = $this->extractMedicineName($groundTruth);
                $predName = $this->extractMedicineName($predicted);
                $predLexName = $predictedLex ? $this->extractMedicineName($predictedLex) : null;
                $isCorrectName = strcasecmp($gtName, $predName) === 0;
                $isCorrectLexName = $predLexName ? strcasecmp($gtName, $predLexName) === 0 : null;

                // Determine which folder to use based on line number
                $lineNumber = intval(trim($row[0]));
                $imageFolder = $lineNumber <= 288 ? 'test' : 'Processed_New';

                $result = [
                    'no' => trim($row[0]),
                    'image_name' => $imageName,
                    'ground_truth' => $groundTruth,
                    'predicted_label' => $predicted,
                    'is_correct' => $isCorrect,
                    'is_correct_name' => $isCorrectName,
                    'image_path' => $this->getImagePath($imageName, $imageFolder),
                ];

                if ($hasLexicon) {
                    $result['predicted_label_lex'] = $predictedLex;
                    $result['is_correct_lex'] = $isCorrectLex;
                    $result['is_correct_lex_name'] = $isCorrectLexName;
                }

                $results[] = $result;
            }
            $currentLine++;
        }
        fclose($file);

        return $results;
    }

    private function getImagePath($imageName, $folder = 'Processed_New')
    {
        // Extract just the filename from paths like "test/dt[0]"
        $filename = basename($imageName);
        
        $baseDir = 'results/' . $folder . '/';
        $imagePath = $baseDir . $filename;

        // Check if the image exists
        if (file_exists(public_path($imagePath))) {
            return asset($imagePath);
        }

        // If not found, try with various extensions
        $extensions = ['.png', '.jpg', '.jpeg', '.gif'];
        $baseNameWithoutExt = pathinfo($filename, PATHINFO_FILENAME);
        
        foreach ($extensions as $ext) {
            $pathWithExt = $baseDir . $baseNameWithoutExt . $ext;
            if (file_exists(public_path($pathWithExt))) {
                return asset($pathWithExt);
            }
        }

        // Fallback: return the attempted path
        return asset($imagePath);
    }

    private function extractMedicineName($text)
    {
        $text = trim($text);
        if ($text === '') {
            return '';
        }

        $parts = explode(':', $text, 2);
        return trim($parts[0]);
    }
}
