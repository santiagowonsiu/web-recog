console.log("I am running");

fetch('/views/objects3d')
    .then(response => response.json())
    .then(data => {
        console.log(data[0].x); // This will print the entire response data
    })
    .catch(error => console.error('Error:', error));

