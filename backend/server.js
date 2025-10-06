// Import required modules
const express = require('express');
const path = require('path');

// Initialize the Express application
const app = express();
const port = 3000;

// --- Middleware ---
app.use(express.urlencoded({ extended: true }));
app.use('/static', express.static(path.join(__dirname, '../frontend/static')));


// --- Routes ---
// Route for the login page
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, '../frontend/templates/login.html'));
});

// NEW: Route for the registration page
app.get('/register', (req, res) => {
    res.sendFile(path.join(__dirname, '../frontend/templates/register.html'));
});

// Route to handle login form submission
app.post('/login', (req, res) => {
    const { email, password } = req.body;
    console.log('--- Login Attempt ---');
    console.log('Email:', email);
    // In a real app, you would check the database
    res.send('Login successful!');
});

// Route to handle registration form submission
app.post('/register', (req, res) => {
    const { name, email, password } = req.body;
    console.log('--- New Registration ---');
    console.log('Name:', name);
    console.log('Email:', email);
    // In a real app, you would save this to a database
    res.send('Registration successful!');
});


// --- Server Startup ---
app.listen(port, () => {
    console.log(`Server is running at http://localhost:${port}`);
});
