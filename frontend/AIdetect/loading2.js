function startLoading() {
    const loadingButton = document.getElementById("loading-button");
    const loadingOverlay = document.getElementById("loading-overlay");
    // const loadingBar = document.getElementsByClassName("progress-bar");
    const loadingBar = document.getElementById("loading-bar");
    const loadingBarMax = document.getElementById("loading-bar-full");

    // Disable the button
    loadingButton.disabled = true;

    // Show the loading overlay
    loadingOverlay.style.display = "flex";

    // Simulate loading (you can replace this with your actual loading logic)
    const progressInterval = setInterval(function () {
        if (loadingBar.clientWidth < loadingBarMax.clientWidth * 0.95) {
            loadingBar.style.width = (loadingBar.clientWidth + 10) + "px";
        } else {
            // Loading is complete
            clearInterval(progressInterval);

            // Re-enable the button
            loadingButton.disabled = false;

            // Hide the loading overlay
            loadingOverlay.style.display = "none";

            // Reset the progress bar width
            loadingBar.style.width = "0";

            //window.location.href = "result.html"
        }
    }, 100); // Adjust the interval and loading duration as needed
    // setInterval(() => {
    //     const computedStyle = getComputedStyle(loadingBar)
    //     const width = parseFloat(computedStyle.getPropertyValue('--width')) || 0
    //     loadingBar.style.setProperty('--width', width + .1)
    // }, 5)
}