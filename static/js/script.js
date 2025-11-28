document.addEventListener('DOMContentLoaded', function() {
    // Check authentication
    if (sessionStorage.getItem('isLoggedIn') !== 'true') {
        window.location.href = 'login.html';
        return;
    }
    
    // DOM Elements
    const cameraSelect = document.getElementById('cameraSelect');
    const toggleDetectionBtn = document.getElementById('toggleDetection');
    // const pauseResumeBtn = document.getElementById('pauseResume'); // Removed
    // const snapshotBtn = document.getElementById('snapshot'); // Removed
    const currentCamera = document.getElementById('currentCamera');
    const videoPlaceholder = document.getElementById('videoPlaceholder');
    const videoStream = document.getElementById('videoStream');
    const fullscreenBtn = document.getElementById('fullscreenBtn');
    const detectionOverlay = document.getElementById('detectionOverlay');
    const peopleCount = document.getElementById('peopleCount');
    const detectionAccuracy = document.getElementById('detectionAccuracy');
    const alertsCount = document.getElementById('alertsCount');
    const personThreshold = document.getElementById('personThreshold');
    const personThresholdValue = document.getElementById('personThresholdValue');
    const helmetThreshold = document.getElementById('helmetThreshold');
    const helmetThresholdValue = document.getElementById('helmetThresholdValue');
    const vestThreshold = document.getElementById('vestThreshold');
    const vestThresholdValue = document.getElementById('vestThresholdValue');
    const confirmationFrames = document.getElementById('confirmationFrames');
    const confirmationFramesValue = document.getElementById('confirmationFramesValue');
    const activityList = document.getElementById('activityList');
    const logoutBtn = document.getElementById('logoutBtn');
    const cameraModal = document.getElementById('cameraModal');
    const cameraForm = document.getElementById('cameraForm');
    const closeModal = document.querySelector('.close-modal');
    const cameraManageBtn = document.getElementById('cameraManageBtn');
    const addCameraBtn = document.getElementById('addCameraBtn');
    const removeCameraBtn = document.getElementById('removeCameraBtn');
    
    // State variables
    let isDetectionActive = false;
    let alertCounter = 2;
    let statsTimer = null;
    let isPaused = false;
    let isInRemoveMode = false;
    
    // Logout handler
    if (logoutBtn) {
        logoutBtn.addEventListener('click', function() {
            sessionStorage.removeItem('isLoggedIn');
            sessionStorage.removeItem('username');
            window.location.href = 'login.html';
        });
    }
    
    // Auto-start: Select first camera and start detection automatically
    setTimeout(() => {
        cameraSelect.value = 'input1'; // Select first camera
        cameraSelect.dispatchEvent(new Event('change')); // Trigger change event
        setTimeout(() => {
            if (isDetectionActive) {
                addActivity('Auto-started: Detection activated');
            }
        }, 800);
    }, 500);
    
    // Camera selection handler
    cameraSelect.addEventListener('change', function() {
        const selectedCamera = cameraSelect.value;
        
        if (selectedCamera) {
            currentCamera.textContent = cameraSelect.options[cameraSelect.selectedIndex].text;
            videoPlaceholder.style.display = 'none';
            videoStream.style.display = 'block';
            
            // Auto-start detection when camera is selected
            if (!isDetectionActive) {
                isDetectionActive = true;
                toggleDetectionBtn.textContent = 'Stop Detection';
            }
            startStream();
            
            // Add activity log
            addActivity(`Camera switched to ${cameraSelect.options[cameraSelect.selectedIndex].text}`);
        } else {
            currentCamera.textContent = 'No Camera Selected';
            videoPlaceholder.style.display = 'flex';
            videoStream.style.display = 'none';
            stopStream();
        }
    });

    // Fullscreen handler
    fullscreenBtn.addEventListener('click', function() {
        const container = document.querySelector('.video-container');
        const el = container || videoStream;
        if (!document.fullscreenElement) {
            if (el.requestFullscreen) el.requestFullscreen();
        } else {
            if (document.exitFullscreen) document.exitFullscreen();
        }
    });

    // Toggle detection handler
    toggleDetectionBtn.addEventListener('click', function() {
        isDetectionActive = !isDetectionActive;
        
        if (isDetectionActive) {
            toggleDetectionBtn.textContent = 'Stop Detection';
            startStream();
            addActivity('People detection activated');
        } else {
            toggleDetectionBtn.textContent = 'Start Detection';
            stopStream();
            addActivity('People detection deactivated');
        }
    });
    
    // Pause/Resume handler - REMOVED
    // pauseResumeBtn.addEventListener('click', function() {
    //     if (!isDetectionActive) return;
    //     isPaused = !isPaused;
    //     if (isPaused) {
    //         pauseResumeBtn.textContent = 'Resume';
    //         stopStats();
    //         videoStream.style.opacity = '0.5';
    //     } else {
    //         pauseResumeBtn.textContent = 'Pause';
    //         startStats();
    //         videoStream.style.opacity = '1';
    //     }
    // });
    
    // Person threshold slider handler
    personThreshold.addEventListener('input', function() {
        personThresholdValue.textContent = (personThreshold.value / 100).toFixed(2);
        if (isDetectionActive && cameraSelect.value) {
            startStream();
        }
    });
    
    // Helmet threshold slider handler
    helmetThreshold.addEventListener('input', function() {
        helmetThresholdValue.textContent = (helmetThreshold.value / 100).toFixed(2);
        if (isDetectionActive && cameraSelect.value) {
            startStream();
        }
    });
    
    // Vest threshold slider handler
    vestThreshold.addEventListener('input', function() {
        vestThresholdValue.textContent = (vestThreshold.value / 100).toFixed(2);
        if (isDetectionActive && cameraSelect.value) {
            startStream();
        }
    });
    
    // Confirmation frames slider handler
    confirmationFrames.addEventListener('input', function() {
        confirmationFramesValue.textContent = confirmationFrames.value;
        if (isDetectionActive && cameraSelect.value) {
            startStream();
        }
    });
    
    // Modal handlers
    if (closeModal) {
        closeModal.addEventListener('click', () => {
            cameraModal.style.display = 'none';
        });
    }
    
    // Add Camera Button
    addCameraBtn.addEventListener('click', function() {
        cameraModal.style.display = 'block';
    });

    // Camera form submission
    cameraForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const cameraName = document.getElementById('cameraName').value;
        const cameraIp = document.getElementById('cameraIp').value;
        const cameraPort = document.getElementById('cameraPort').value;
        const cameraUsername = document.getElementById('cameraUsername').value;
        const cameraPassword = document.getElementById('cameraPassword').value;
        
        // Add new camera to dropdown
        const option = document.createElement('option');
        const cameraId = `custom_${Date.now()}`;
        option.value = cameraId;
        option.textContent = cameraName;
        
        // Store camera details in dataset
        option.dataset.ip = cameraIp;
        option.dataset.port = cameraPort;
        option.dataset.username = cameraUsername;
        option.dataset.password = cameraPassword;
        
        cameraSelect.appendChild(option);
        
        // Select the newly added camera
        cameraSelect.value = cameraId;
        cameraSelect.dispatchEvent(new Event('change'));
        
        // Close modal and reset form
        cameraModal.style.display = 'none';
        this.reset();
        
        addActivity(`Added new camera: ${cameraName}`);
    });
    
    // Remove Camera Button
    removeCameraBtn.addEventListener('click', function() {
        const selectedIndex = cameraSelect.selectedIndex;
        if (selectedIndex >= 0) {
            const selectedOption = cameraSelect.options[selectedIndex];
            if (selectedOption && (selectedOption.value || selectedOption.dataset.ip)) {
                const cameraName = selectedOption.textContent;
                if (confirm(`Are you sure you want to remove ${cameraName}?`)) {
                    // Remove from dropdown
                    cameraSelect.remove(selectedIndex);
                    addActivity(`Removed camera: ${cameraName}`);
                    
                    // Select another camera if available
                    if (cameraSelect.options.length > 0) {
                        cameraSelect.selectedIndex = Math.min(selectedIndex, cameraSelect.options.length - 1);
                        cameraSelect.dispatchEvent(new Event('change'));
                    } else {
                        // No cameras left
                        removeCameraBtn.disabled = true;
                        currentCamera.textContent = 'No Camera Selected';
                        videoPlaceholder.style.display = 'flex';
                        videoStream.style.display = 'none';
                        stopStream();
                    }
                }
            } else {
                alert('Please select a valid camera to remove');
            }
        } else {
            alert('No camera selected for removal');
        }
    });

    // Combined camera management button
    cameraManageBtn.addEventListener('click', function() {
        if (!isInRemoveMode) {
            // Add Camera mode - open modal
            cameraModal.style.display = 'block';
        } else {
            // Remove Camera mode
            const selectedIndex = cameraSelect.selectedIndex;
            if (selectedIndex >= 0) {
                const selectedOption = cameraSelect.options[selectedIndex];
                if (selectedOption && (selectedOption.value || selectedOption.dataset.ip)) {
                    const cameraName = selectedOption.textContent;
                    if (confirm(`Are you sure you want to remove ${cameraName}?`)) {
                        // Remove from dropdown
                        cameraSelect.remove(selectedIndex);
                        addActivity(`Removed camera: ${cameraName}`);

                        // Select another camera if available
                        if (cameraSelect.options.length > 0) {
                            cameraSelect.selectedIndex = Math.min(selectedIndex, cameraSelect.options.length - 1);
                            cameraSelect.dispatchEvent(new Event('change'));
                        } else {
                            // No cameras left
                            currentCamera.textContent = 'No Camera Selected';
                            videoPlaceholder.style.display = 'flex';
                            videoStream.style.display = 'none';
                            stopStream();
                        }
                    }
                } else {
                    alert('Please select a valid camera to remove');
                }
            } else {
                alert('No camera selected for removal');
            }
        }
    });

    // Toggle between add/remove modes on button double click
    cameraManageBtn.addEventListener('dblclick', function() {
        isInRemoveMode = !isInRemoveMode;
        updateManageButton();
    });

    function updateManageButton() {
        if (isInRemoveMode) {
            cameraManageBtn.innerHTML = `
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M19 7L18.1327 19.1425C18.0579 20.1891 17.187 21 16.1378 21H7.86224C6.81296 21 5.94208 20.1891 5.86732 19.1425L5 7M10 11V17M14 11V17M15 7V4C15 3.44772 14.5523 3 14 3H10C9.44772 3 9 3.44772 9 4V7M4 7H20" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
                <span>Remove Camera</span>
            `;
            cameraManageBtn.classList.add('remove-mode');
        } else {
            cameraManageBtn.innerHTML = `
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M12 4V20M4 12H20" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
                <span>Add Camera</span>
            `;
            cameraManageBtn.classList.remove('remove-mode');
        }
    }

    // Initialize button state
    updateManageButton();

    // Keyboard shortcut handler
    document.addEventListener('keydown', function(e) {
        if (e.key === '+' || e.key === '=') {
            // Add camera
            cameraModal.style.display = 'block';
            cameraManageBtn.classList.remove('remove-mode');
        } else if (e.key === '-' || e.key === '_') {
            // Remove camera
            const selectedOption = cameraSelect.options[cameraSelect.selectedIndex];
            if (selectedOption && selectedOption.value) {
                const cameraName = selectedOption.textContent;
                if (confirm(`Are you sure you want to remove ${cameraName}?`)) {
                    selectedOption.remove();
                    addActivity(`Removed camera: ${cameraName}`);
                    
                    // Select first camera if available
                    if (cameraSelect.options.length > 1) {
                        cameraSelect.selectedIndex = 1;
                    } else {
                        cameraSelect.selectedIndex = 0;
                    }
                    cameraSelect.dispatchEvent(new Event('change'));
                }
            }
            cameraManageBtn.classList.add('remove-mode');
        }
    });
    
    // Build the stream URL with current settings
    function buildStreamURL() {
        const selectedCamera = cameraSelect.value;
        if (!selectedCamera) return '';
        
        // Get threshold values from sliders
        const personConf = (personThreshold.value / 100).toFixed(2);
        const helmetConf = (helmetThreshold.value / 100).toFixed(2);
        const vestConf = (vestThreshold.value / 100).toFixed(2);
        const frames = confirmationFrames.value;
        
        const params = new URLSearchParams({ 
            source: selectedCamera, 
            person_conf: personConf,
            helmet_conf: helmetConf,
            vest_conf: vestConf,
            confirmation_frames: frames,
            w: '640', 
            skip: '1' 
        });
        return `/video_feed?${params.toString()}`;
    }

    function startStream() {
        if (!cameraSelect.value) return;
        
        // Reset quick stats
        peopleCount.textContent = '—';
        detectionAccuracy.textContent = '—';
        
        const videoUrl = buildStreamURL();
        
        // Use img element for MJPEG stream with YOLOv8 detection
        videoStream.style.display = 'none';
        
        // Remove old detection stream if exists
        let detectionStream = document.getElementById('detectionStream');
        if (detectionStream) {
            detectionStream.remove();
        }
        
        // Create new img element for detection stream
        detectionStream = document.createElement('img');
        detectionStream.id = 'detectionStream';
        detectionStream.style.width = '100%';
        detectionStream.style.height = '100%';
        detectionStream.style.objectFit = 'contain';
        detectionStream.src = videoUrl;
        
        const videoFeed = document.querySelector('.video-feed');
        videoFeed.insertBefore(detectionStream, videoFeed.firstChild);
        
        detectionStream.onerror = () => {
            addActivity('Error loading detection stream - Make sure YOLOv8 is running');
            console.error('Detection stream error');
        };
        
        isPaused = false;
        // pauseResumeBtn.textContent = 'Pause'; // Removed
        startStats();
    }

    function stopStream() {
        stopStats();
        videoStream.removeAttribute('src');
        videoStream.style.display = 'none';
        
        // Remove detection stream
        const detectionStream = document.getElementById('detectionStream');
        if (detectionStream) {
            detectionStream.src = '';
            detectionStream.remove();
        }
        
        peopleCount.textContent = '0';
        detectionAccuracy.textContent = '0%';
        document.getElementById('avgResponse').textContent = '—';
    }

    function startStats() {
        stopStats();
        statsTimer = setInterval(async () => {
            try {
                const res = await fetch('/stats');
                if (!res.ok) return;
                const data = await res.json();
                
                const totalPeople = data.count || 0;
                const noHelmetCount = data.no_helmet || 0;
                const noVestCount = data.no_vest || 0;
                
                peopleCount.textContent = totalPeople;
                
                // Only generate alerts for safety violations
                if (noHelmetCount > 0 || noVestCount > 0) {
                    alertCounter++;
                    alertsCount.textContent = alertCounter;
                    
                    let alertMessage = 'Safety Alert: ';
                    if (noHelmetCount > 0 && noVestCount > 0) {
                        alertMessage += `${noHelmetCount} without helmet and ${noVestCount} without vest`;
                    } else if (noHelmetCount > 0) {
                        alertMessage += `${noHelmetCount} without helmet`;
                    } else {
                        alertMessage += `${noVestCount} without vest`;
                    }
                    
                    addActivity(alertMessage);
                }
            } catch (e) {
                console.error('Error fetching stats:', e);
            }
        }, 1000);
    }

    function stopStats() {
        if (statsTimer) {
            clearInterval(statsTimer);
            statsTimer = null;
        }
    }
    
    // Add activity to the log
    function addActivity(message) {
        const now = new Date();
        const timeString = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        
        const activityItem = document.createElement('li');
        activityItem.className = 'activity-item';
        
        activityItem.innerHTML = `
            <div class="activity-icon">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M12 8V12L15 15M21 12C21 16.9706 16.9706 21 12 21C7.02944 21 3 16.9706 3 12C3 7.02944 7.02944 3 12 3C16.9706 3 21 7.02944 21 12Z" stroke="#3b82f6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
            </div>
            <div class="activity-details">
                <div>${message}</div>
                <div class="activity-time">${timeString}</div>
            </div>
        `;
        
        activityList.insertBefore(activityItem, activityList.firstChild);
        
        // Limit to 10 activities
        if (activityList.children.length > 10) {
            activityList.removeChild(activityList.lastChild);
        }
    }
});