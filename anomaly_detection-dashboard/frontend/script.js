document.getElementById('fetchData').addEventListener('click', async () => {
    const response = await fetch('http://127.0.0.1:5000/detect');
    const data = await response.json();
    displayResults(data);
});

function displayResults(data) {
    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = '<h2>Detected Anomalies:</h2>';
    data.forEach((item, index) => {
        resultsDiv.innerHTML += `
            <div class="anomaly">
                <p><strong>Anomaly ${index + 1}:</strong></p>
                <p>Date: ${item.Date}</p>
                <p>Close: ${item.Close}</p>
                <p>Returns: ${item.returns}</p>
            </div>
        `;
    });
}