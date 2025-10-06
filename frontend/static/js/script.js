const signUpButton = document.getElementById('signUp');
const signInButton = document.getElementById('signIn');
const container = document.getElementById('container');

// Event listener for the "Sign Up" button
signUpButton.addEventListener('click', () => {
	container.classList.add("right-panel-active");
});

// Event listener for the "Sign In" button
signInButton.addEventListener('click', () => {
	container.classList.remove("right-panel-active");
});
