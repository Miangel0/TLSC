const toggleButton = document.getElementById('toggle-camera');
const video = document.getElementById('camera-video');
const placeholder = document.getElementById('camera-placeholder');
let stream = null;

toggleButton.addEventListener('click', async () => {
    if (!stream) {
        try {
            // Pedir acceso a la cámara
            stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;

            // Mostrar video y ocultar imagen
            video.style.display = 'block';
            placeholder.style.display = 'none';

            // Cambiar texto y ancho del botón
            toggleButton.textContent = "Desactivar cámara";
            toggleButton.classList.add("wide");

            // Mover botón debajo del video (opcional)
            toggleButton.style.marginTop = '10px';

        } catch (err) {
            alert('No se pudo acceder a la cámara: ' + err);
        }
    } else {
        // Detener la cámara
        stream.getTracks().forEach(track => track.stop());
        stream = null;

        // Ocultar video y mostrar imagen
        video.style.display = 'none';
        placeholder.style.display = 'block';

        // Cambiar texto y volver al tamaño original
        toggleButton.textContent = "Activar Cámara";
        toggleButton.classList.remove("wide");
    }
});
