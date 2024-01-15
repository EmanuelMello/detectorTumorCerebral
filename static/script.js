document.getElementById('file-input').addEventListener('change', function() {
    var fileName = this.value.split('\\').pop();
    document.getElementById('file-label').innerText = fileName || 'Escolha um arquivo';
});

document.getElementById('upload-form').addEventListener('submit', function() {
    document.getElementById('submit-button').innerText = 'Enviando...';
});
