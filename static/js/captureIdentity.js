document.addEventListener('DOMContentLoaded', function() {
    console.log("El DOM está completamente cargado y analizado.");
    // Usar el nombre de usuario pasado desde la plantilla
    console.log("Nombre capturado:", userName); // Para depurar
    if (userName) {
        const newSrc = `/video_feed?name=${encodeURIComponent(userName)}`;
        console.log("Nueva URL:", newSrc); // Para depurar
        document.getElementById('bg').src = newSrc;
    } else {
        alert('Por favor, ingrese un nombre.');
    }
});
