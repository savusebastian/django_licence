$(document).ready(() => {
    const chooseBtn = document.getElementById('image-file');
    const selectBtn = document.getElementById('select-btn');
    const submitBtn = document.getElementById('submit-btn');
    const uploadBtn = document.getElementById('upload-btn');
    const customTxt = document.getElementById('custom-span');
    const customSize = document.getElementById('custom-size');

    // Change the action when clicking the input button to the custom button
    selectBtn.addEventListener('click', function(){
        chooseBtn.click();
    });

    uploadBtn.addEventListener('click', function(){
        submitBtn.click();
    });

    // Function for getting image size
    function getFileSize(file){
        var file = file.files[0];
        if(file.size > (1024 * 1024)) return Math.round(file.size / (1024 * 1024)) + ' MB';
        if(file.size > 1024) return Math.round(file.size / 1024) + ' KB';
        return file.size + ' B';
    };

    chooseBtn.addEventListener('change', function(){
        const preview = document.querySelector('.image-upload');
        const file = document.querySelector('input[type=file]').files[0];
        const reader = new FileReader();

        reader.addEventListener('load', function(){
            // convert image file to base64 string
            preview.src = reader.result;
        }, false);

        if(file){
            reader.readAsDataURL(file);
        }
    });

    chooseBtn.addEventListener('change', function() {
        if(chooseBtn.value){
            var txt = chooseBtn.value.split('\\').pop().split('/').pop();
            customTxt.innerHTML = txt;
            customSize.innerHTML = getFileSize(chooseBtn);
        }
        else{
            customTxt.innerHTML = 'No file chosen';
        }
    });
});


// Return filename + extension from a input type = file
// chooseBtn.value.split('\\').pop().split('/').pop()
// Return filename + extension from a given path
// console.log(path.split('/').pop())
