const toggleButton = document.getElementById('toggle-camera');
const placeholder = document.getElementById('camera-placeholder');
const streamImg = document.getElementById('camera-stream');

let cameraActive = false;

toggleButton.addEventListener('click', async () => {
    if (!cameraActive) {
        // Activar cámara → pedir el stream al backend
        streamImg.src = "http://localhost:5000/video_feed";
        streamImg.style.display = 'block';
        placeholder.style.display = 'none';

        toggleButton.textContent = "Desactivar cámara";
        toggleButton.classList.add("wide");
        cameraActive = true;
    } else {
        // Desactivar cámara → cortar la conexión
        streamImg.src = ""; // Detiene el request
        streamImg.style.display = 'none';
        placeholder.style.display = 'block';

        toggleButton.textContent = "Activar cámara";
        toggleButton.classList.remove("wide");
        cameraActive = false;
    }
});
