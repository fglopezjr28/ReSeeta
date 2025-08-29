<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ReSeeta</title>
</head>
<body>

    @auth
    <p>Congratulations! You are logged in!</p>
    <form action="/logout" method="POST">
        @csrf
        <button type="submit">Logout</button>
    </form>

    <div>
        <h2>Create a Post<h2>   
            <form action="/create-post" method="POST">
                @csrf
                <input name="title" type = "text" placeholder="Title">
                <textarea name="body" placeholder="Body"></textarea>
                <button type = "submit">Create Post</button>
        </form>
    </div>
    <div>
        <h2>All Posts<h2>   
        @foreach ($posts as $post)
        <div>
            <h3>{{ $post['title'] }} by {{ $post->user->name }}</h3>
            <p>{{ $post['body'] }}</p>
            <p><a href="/edit-post/{{ $post->id }}">Edit</a></p>
            <form action="/delete-post/{{ $post->id }}" method="POST">
                @csrf
                @method('DELETE')
                <button type="submit">Delete</button>
        </div>
        @endforeach
    </div>    
    @else
        <div>
        <h2>Register<h2>   
            <form action="/register" method="POST">
                @csrf
                <input name="name" type = "text" placeholder="Name">
                <input name="email" type = "text" placeholder="Email">
                <input name="password" type = "password" placeholder="Password">
                <button type = "submit">Register</button>
        </form>
        </div>
        <div>
        <h2>Login<h2>   
            <form action="/login" method="POST">
                @csrf
                <input name="loginname" type = "text" placeholder="Name">
                <input name="loginpassword" type = "password" placeholder="Password">
                <button type = "submit">Login</button>
        </form>
        </div>
    @endauth
    
</body>
</html>