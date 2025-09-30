<?php

use App\Models\Post;
use Illuminate\Support\Facades\Route;
use App\Http\Controllers\PostController;
use App\Http\Controllers\UserController;
use App\Http\Controllers\OcrController;

/*
|--------------------------------------------------------------------------
| Web Routes
|--------------------------------------------------------------------------
|
| Here is where you can register web routes for your application. These
| routes are loaded by the RouteServiceProvider within a group which
| contains the "web" middleware group. Now create something great!
|
*/

Route::get('/', function () {
    return view('landing');
});

Route::get('/about', function () {
    return view('about');
});

Route::get('/convert', function () {
    return view('convert');
});

// Route::get('/', function () {
//     $posts = [];
//     if (auth()->check()) {
//         $posts = auth()->user()->Posts()->latest()->get();
//     }
//     //$posts = Post::where('user_id', auth()->id())->get();
//     return view('home', ['posts' => $posts]);
// });

//User Info Routes
Route::post('/register', [UserController::class, 'register']);
Route::post('logout', [UserController::class, 'logout']);
Route::post('login', [UserController::class, 'login']);

//Blog Routes
Route::post('/create-post', [PostController::class, 'createPost']);
Route::get('/edit-post/{post}', [PostController::class, 'showEditScreen']);
Route::put('/edit-post/{post}', [PostController::class, 'acutallyUpdatePost']);
Route::delete('/delete-post/{post}', [PostController::class, 'deletePost']);

Route::post('/ocr/predict', [OcrController::class, 'predict'])->name('ocr.predict');