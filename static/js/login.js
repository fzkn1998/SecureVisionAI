document.addEventListener('DOMContentLoaded', function() {
    const loginForm = document.getElementById('loginForm');
    const errorMessage = document.getElementById('errorMessage');
    const loginBtn = loginForm.querySelector('.login-btn');
    
    // Check if user is already logged in
    if (sessionStorage.getItem('isLoggedIn') === 'true') {
        window.location.href = 'index.html';
        return;
    }
    
    // Default credentials (in production, this should be handled server-side)
    const validCredentials = {
        username: 'admin',
        password: 'admin123'
    };
    
    loginForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const username = document.getElementById('username').value.trim();
        const password = document.getElementById('password').value;
        const rememberMe = document.getElementById('rememberMe').checked;
        
        // Clear previous error
        errorMessage.classList.remove('show');
        
        // Show loading state
        loginBtn.classList.add('loading');
        loginBtn.querySelector('span').textContent = 'Signing in';
        
        // Simulate authentication delay
        setTimeout(() => {
            // Validate credentials
            if (username === validCredentials.username && password === validCredentials.password) {
                // Successful login
                sessionStorage.setItem('isLoggedIn', 'true');
                sessionStorage.setItem('username', username);
                
                if (rememberMe) {
                    localStorage.setItem('rememberUser', username);
                }
                
                // Show success and redirect
                loginBtn.querySelector('span').textContent = 'Success!';
                setTimeout(() => {
                    window.location.href = 'index.html';
                }, 500);
            } else {
                // Failed login
                loginBtn.classList.remove('loading');
                loginBtn.querySelector('span').textContent = 'Sign In';
                
                errorMessage.textContent = 'Invalid username or password. Please try again.';
                errorMessage.classList.add('show');
                
                // Shake the form
                loginForm.style.animation = 'shake 0.5s ease';
                setTimeout(() => {
                    loginForm.style.animation = '';
                }, 500);
            }
        }, 1000);
    });
    
    // Auto-fill remembered username
    const rememberedUser = localStorage.getItem('rememberUser');
    if (rememberedUser) {
        document.getElementById('username').value = rememberedUser;
        document.getElementById('rememberMe').checked = true;
    }
    
    // Handle forgot password click
    document.querySelector('.forgot-password').addEventListener('click', function(e) {
        e.preventDefault();
        alert('Please contact your system administrator to reset your password.');
    });
});
