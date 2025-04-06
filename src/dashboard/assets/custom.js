/* Custom JavaScript for StockPred Dashboard */

// This script helps ensure the dashboard loads properly
window.addEventListener('DOMContentLoaded', (event) => {
    console.log('Dashboard DOM fully loaded');
    
    // Add connection status indicator
    const statusDiv = document.createElement('div');
    statusDiv.id = 'connection-status';
    statusDiv.style.position = 'fixed';
    statusDiv.style.bottom = '10px';
    statusDiv.style.right = '10px';
    statusDiv.style.padding = '5px 10px';
    statusDiv.style.borderRadius = '5px';
    statusDiv.style.backgroundColor = '#27ae60';
    statusDiv.style.color = 'white';
    statusDiv.style.fontSize = '12px';
    statusDiv.style.zIndex = '1000';
    statusDiv.textContent = 'Connected';
    document.body.appendChild(statusDiv);
    
    // Set up connection monitoring
    let connectionLost = false;
    
    function checkConnection() {
        const status = document.getElementById('connection-status');
        
        if (navigator.onLine && !connectionLost) {
            status.style.backgroundColor = '#27ae60';
            status.textContent = 'Connected';
        } else {
            status.style.backgroundColor = '#e74c3c';
            status.textContent = 'Disconnected';
            
            // Try to reconnect
            if (connectionLost) {
                window.location.reload();
            }
        }
    }
    
    // Check connection status regularly
    setInterval(checkConnection, 5000);
    
    // Handle online/offline events
    window.addEventListener('online', function() {
        const status = document.getElementById('connection-status');
        status.style.backgroundColor = '#27ae60';
        status.textContent = 'Connected';
        connectionLost = false;
    });
    
    window.addEventListener('offline', function() {
        const status = document.getElementById('connection-status');
        status.style.backgroundColor = '#e74c3c';
        status.textContent = 'Disconnected';
        connectionLost = true;
    });
}); 