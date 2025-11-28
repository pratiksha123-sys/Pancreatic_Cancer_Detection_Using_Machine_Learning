// static/script.js

document.addEventListener('DOMContentLoaded', () => {
    // Get references to all navigation buttons
    const navHome = document.getElementById('nav-home');
    const navLogin = document.getElementById('nav-login');
    const navAbout = document.getElementById('nav-about');
    const navDetection = document.getElementById('nav-detection');
    const navComparison = document.getElementById('nav-comparison');
    const navAnalysis = document.getElementById('nav-analysis'); // New nav button for Analysis
    const navLogout = document.getElementById('nav-logout');

    // Get references to all main content sections
    const homeSection = document.getElementById('home-section');
    const authSection = document.getElementById('auth-section');
    const aboutSection = document.getElementById('about-section');
    const detectionSection = document.getElementById('detection-section');
    const comparisonSection = document.getElementById('comparison-section');
    const analysisSection = document.getElementById('analysis-section'); // New section for Analysis
    const homeExtra = document.getElementById('home-extra'); // new wrapper for services/about/department
    const homeRightBox = document.getElementById('home-right-box');

    


    // Get references for authentication forms
    const authTitle = document.getElementById('auth-title');
    const loginForm = document.getElementById('login-form');
    const registerForm = document.getElementById('register-form');
    const toggleAuthText = document.getElementById('toggle-auth-text');
    const toggleAuthBtn = document.getElementById('toggle-auth-btn');
    const authMessage = document.getElementById('auth-message');

    // Get references for home page buttons
    const homeLoginBtn = document.getElementById('home-login-btn');
    const homeRegisterBtn = document.getElementById('home-register-btn');

    // Get references for detection page elements
    const detectionForm = document.getElementById('detection-form');
    const detectionResult = document.getElementById('detection-result');
    const resultMessage = document.getElementById('result-message');
    const resultProbability = document.getElementById('result-probability');
    const detectionMessage = document.getElementById('detection-message');

    // Get references for comparison page elements
    const comparisonData = document.getElementById('comparison-data');
    const comparisonMessage = document.getElementById('comparison-message');
    const comparisonLoadingMessage = document.getElementById('comparison-loading-message');

    // Get references for analysis page elements
    const totalRecordsSpan = document.getElementById('total-records');
    const cancerCasesSummarySpan = document.getElementById('cancer-cases-summary');
    const nonCancerCasesSummarySpan = document.getElementById('non-cancer-cases-summary');
    const cancerPercentageSummarySpan = document.getElementById('cancer-percentage-summary');
    const analysisMessage = document.getElementById('analysis-message');

    let loggedInUser = null; // Variable to store the logged-in username (for frontend state)
    let charts = {}; // Object to store Chart.js instances

    // --- Helper Functions ---

    /**
     * Hides all main content sections and shows only the specified one.
     * @param {HTMLElement} section - The section element to show.
     */
    function showSection(section) {
    // Hide all main sections
    homeSection.classList.add('hidden');
    authSection.classList.add('hidden');
    detectionSection.classList.add('hidden');
    comparisonSection.classList.add('hidden');
    analysisSection.classList.add('hidden');
    homeExtra.classList.add('hidden');
    
    // Show the requested section
    section.classList.remove('hidden');

    // Show extras only for home section
    if (section === homeSection) {
        homeExtra.classList.remove('hidden');
    }
}



    /**
     * Updates the visibility of navigation links based on login status.
     */
    function updateNavVisibility() {
        if (loggedInUser) {
            // If logged in, hide Login/Register and show Detection, Comparison, Analysis, Logout
            if (homeRightBox) homeRightBox.classList.add('invisible-box');
            navLogin.classList.add('hidden');
            navAbout.classList.remove('hidden');
            navDetection.classList.remove('hidden');
            navComparison.classList.remove('hidden');
            navAnalysis.classList.remove('hidden'); // Show Analysis button
            navLogout.classList.remove('hidden');
        } else {
            // If logged out, show Login/Register and hide Detection, Comparison, Analysis, Logout
            if (homeRightBox) homeRightBox.classList.remove('invisible-box');
            navLogin.classList.remove('hidden');
            navAbout.classList.add('hidden');
            navDetection.classList.add('hidden');
            navComparison.classList.add('hidden');
            navAnalysis.classList.add('hidden'); // Hide Analysis button
            navLogout.classList.add('hidden');
        }
    }

    /**
     * Displays a temporary message on the UI.
     * @param {HTMLElement} element - The HTML element where the message will be displayed.
     * @param {string} message - The message text.
     * @param {string} type - 'success' or 'error' for styling.
     */
    function showMessage(element, message, type) {
        element.textContent = message;
        element.className = `mt-4 text-center text-sm font-medium message-box ${type}`;
        setTimeout(() => {
            element.textContent = '';
            element.className = 'mt-4 text-center text-sm font-medium'; // Reset class
        }, 3000); // Message disappears after 3 seconds
    }

    // --- Navigation Button Event Listeners ---

    navHome.addEventListener('click', () => {
        showSection(homeSection);
    });

    navLogin.addEventListener('click', () => {
        showSection(authSection);
        authTitle.textContent = 'Login'; // Set title to Login
        loginForm.classList.remove('hidden'); // Show login form
        registerForm.classList.add('hidden'); // Hide register form
        document.getElementById('login-identifier').value = ''; // Clear login fields
        document.getElementById('login-password').value = '';
        toggleAuthText.textContent = "Don't have an account?";
        toggleAuthBtn.textContent = "Register here";
        authMessage.textContent = ''; // Clear any previous auth messages
    });

    navAbout.addEventListener('click', () => {
        showSection(homeSection);
        setTimeout(() => {
            aboutSection.scrollIntoView({ behavior: 'smooth' });
        }, 100);
    });


    navDetection.addEventListener('click', () => {
        showSection(detectionSection);
        // Dynamically create detection form fields when this section is shown
        generateDetectionForm();
    });

    navComparison.addEventListener('click', () => {
        showSection(comparisonSection);
        fetchModelComparisonData(); // Fetch and display model comparison data
    });

    navAnalysis.addEventListener('click', () => { // Event listener for Analysis
        showSection(analysisSection);
        fetchAnalysisData(); // Fetch and render analysis charts
    });

    navLogout.addEventListener('click', async () => {
    try {
        const res = await fetch('/logout', { method: 'POST' });
        if (res.ok) {
            loggedInUser = null;
            updateNavVisibility();
            showSection(homeSection);
            showMessage(authMessage, 'Logged out successfully', 'success');
        } else {
            console.error('Logout failed.');
        }
    } catch (err) {
        console.error('Logout error:', err);
    }
});


    const exploreMoreBtn = document.getElementById('explore-more-btn');
    if (exploreMoreBtn) {
        exploreMoreBtn.addEventListener('click', (e) => {
            e.preventDefault(); // Stop default behavior
            showSection(homeSection); // Ensure home section is shown
            window.scrollTo({ top: 0, behavior: 'smooth' }); // Scroll to top smoothly
        });
    }



    // --- Home Page Button Event Listeners ---
    homeLoginBtn.addEventListener('click', () => {
        showSection(authSection);
        authTitle.textContent = 'Login';
        loginForm.classList.remove('hidden');
        registerForm.classList.add('hidden');
        document.getElementById('login-identifier').value = ''; // Clear login fields
        document.getElementById('login-password').value = '';
        toggleAuthText.textContent = "Don't have an account?";
        toggleAuthBtn.textContent = "Register here";
        authMessage.textContent = '';
    });

    homeRegisterBtn.addEventListener('click', () => {
        showSection(authSection);
        authTitle.textContent = 'Register';
        registerForm.classList.remove('hidden');
        loginForm.classList.add('hidden');
        registerForm.reset(); // Clear register fields
        toggleAuthText.textContent = "Already have an account?";
        toggleAuthBtn.textContent = "Login here";
        authMessage.textContent = '';
    });

    // --- Authentication Section Logic ---

    // Toggle between Login and Register forms
    toggleAuthBtn.addEventListener('click', () => {
        if (loginForm.classList.contains('hidden')) {
            // Currently showing register form, switch to login
            authTitle.textContent = 'Login';
            loginForm.classList.remove('hidden');
            registerForm.classList.add('hidden');
            document.getElementById('login-identifier').value = ''; // Clear login fields
            document.getElementById('login-password').value = '';
            toggleAuthText.textContent = "Don't have an account?";
            toggleAuthBtn.textContent = "Register here";
        } else {
            // Currently showing login form, switch to register
            authTitle.textContent = 'Register';
            registerForm.classList.remove('hidden');
            loginForm.classList.add('hidden');
            registerForm.reset(); // Clear register fields
            toggleAuthText.textContent = "Already have an account?";
            toggleAuthBtn.textContent = "Login here";
        }
        authMessage.textContent = ''; // Clear previous messages
    });

    // Handle Login form submission
    loginForm.addEventListener('submit', async (e) => {
        e.preventDefault(); // Prevent default form submission
        const identifier = document.getElementById('login-identifier').value;
        const password = document.getElementById('login-password').value;

        try {
            const response = await fetch('/login', { // Send login data to backend
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ identifier, password })
            });
            const data = await response.json(); // Parse JSON response
            if (response.ok) {
                loggedInUser = data.username; // Store username on successful login
                showMessage(authMessage, data.message, 'success');
                updateNavVisibility(); // Update nav bar
                showSection(homeSection); // Go back to home after successful login
            } else {
                showMessage(authMessage, data.message, 'error');
            }
        } catch (error) {
            console.error('Login error:', error);
            showMessage(authMessage, 'An error occurred during login. Please check your network connection.', 'error');
        }
    });

    // Handle Register form submission
    registerForm.addEventListener('submit', async (e) => {
        e.preventDefault(); // Prevent default form submission
        const username = document.getElementById('register-username').value;
        const password = document.getElementById('register-password').value;
        const email = document.getElementById('register-email').value;
        const full_name = document.getElementById('register-full-name').value;
        const date_of_birth = document.getElementById('register-dob').value;
        const country = document.getElementById('register-country').value;
        const gender = document.getElementById('register-gender').value;
        const phone_number = document.getElementById('register-phone').value;


        try {
            const response = await fetch('/register', { // Send registration data to backend
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password, email, full_name, date_of_birth, country, gender, phone_number })
            });
            const data = await response.json(); // Parse JSON response
            if (response.ok) {
                showMessage(authMessage, data.message + ' You can now log in.', 'success');
                // Automatically switch to login form after successful registration
                authTitle.textContent = 'Login';
                loginForm.classList.remove('hidden');
                registerForm.classList.add('hidden');
                toggleAuthText.textContent = "Don't have an account?";
                toggleAuthBtn.textContent = "Register here";
                registerForm.reset(); // Clear the registration form fields
            } else {
                showMessage(authMessage, data.message, 'error');
            }
        } catch (error) {
            console.error('Registration error:', error);
            showMessage(authMessage, 'An error occurred during registration. Please check your network connection.', 'error');
        }
    });

    // --- Detection Section Logic ---

    // Define the expected features and their types/options for the detection form.
    // This array MUST EXACTLY MATCH the 'feature_columns' list from your app.py's output.
    const detectionFeatures = [
        { name: 'Country', type: 'select', options: ['United States', 'India', 'Germany', 'Canada', 'South Africa', 'United Kingdom', 'Australia', 'France', 'Japan'] },
        { name: 'Age', type: 'number', min: 0, max: 120 },
        { name: 'Gender', type: 'select', options: ['Male', 'Female'] },
        { name: 'Smoking_History', type: 'select', options: [{ label: 'No', value: '0' },{ label: 'Yes', value: '1' } ]},
        { name: 'Obesity', type: 'select', options: [{ label: 'No', value: '0' },{ label: 'Yes', value: '1' } ] },
        { name: 'Diabetes', type: 'select', options: [{ label: 'No', value: '0' },{ label: 'Yes', value: '1' } ] },
        { name: 'Chronic_Pancreatitis', type: 'select', options: [{ label: 'No', value: '0' },{ label: 'Yes', value: '1' } ] },
        { name: 'Family_History', type: 'select', options: [{ label: 'No', value: '0' },{ label: 'Yes', value: '1' } ] },
        { name: 'Hereditary_Condition', type: 'select', options: [{ label: 'No', value: '0' },{ label: 'Yes', value: '1' } ] },
        { name: 'Jaundice', type: 'select', options: [{ label: 'No', value: '0' },{ label: 'Yes', value: '1' } ] },
        { name: 'Abdominal_Discomfort', type: 'select', options: [{ label: 'No', value: '0' },{ label: 'Yes', value: '1' } ] },
        { name: 'Back_Pain', type: 'select', options: [{ label: 'No', value: '0' },{ label: 'Yes', value: '1' } ] },
        { name: 'Weight_Loss', type: 'select', options: [{ label: 'No', value: '0' },{ label: 'Yes', value: '1' } ] },
        { name: 'Development_of_Type2_Diabetes', type: 'select', options: [{ label: 'No', value: '0' },{ label: 'Yes', value: '1' } ] },
        { name: 'Alcohol_Consumption', type: 'select', options: [{ label: 'No', value: '0' },{ label: 'Yes', value: '1' } ] },
        { name: 'Physical_Activity_Level', type: 'select', options: ['Low', 'Medium', 'High'] },
        { name: 'Diet_Processed_Food', type: 'select', options: ['Low', 'Medium', 'High'] },
    ];


    /**
     * Dynamically generates input fields for the detection form based on `detectionFeatures`.
     */
    function generateDetectionForm() {
        detectionForm.innerHTML = ''; // Clear any existing fields
        if (detectionFeatures.length === 0) {
            detectionForm.innerHTML = '<p class="text-center text-red-500">No detection features defined in script.js. Please update the `detectionFeatures` array based on your backend output.</p>';
            return;
        }

        detectionFeatures.forEach(feature => {
            const div = document.createElement('div');
            div.className = 'form-group';
            const label = document.createElement('label');
            label.htmlFor = feature.name.toLowerCase().replace(/_/g, '-');
            label.className = 'block text-gray-700 text-sm font-semibold mb-2';
            label.textContent = feature.name.replace(/_/g, ' ') + ':';

            if (feature.type === 'number') {
                const input = document.createElement('input');
                input.type = 'number';
                input.id = feature.name.toLowerCase().replace(/_/g, '-');
                input.name = feature.name;
                input.className = 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-emerald-500 focus:border-emerald-500 shadow-sm';
                input.required = true;
                if (feature.min !== undefined) input.min = feature.min;
                if (feature.max !== undefined) input.max = feature.max;
                div.appendChild(label);
                div.appendChild(input);
            } else if (feature.type === 'select') {
                const select = document.createElement('select');
                select.id = feature.name.toLowerCase().replace(/_/g, '-');
                select.name = feature.name;
                select.className = 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-emerald-500 focus:border-emerald-500 shadow-sm';
                select.required = true;
                feature.options.forEach(optionItem => {
                const option = document.createElement('option');

                if (typeof optionItem === 'object' && optionItem !== null) {
                    option.value = optionItem.value;
                    option.textContent = optionItem.label;
                } else {
                    option.value = optionItem;
                    option.textContent = optionItem;
                }

                select.appendChild(option);
            });
                div.appendChild(label);
                div.appendChild(select);
            }
            detectionForm.appendChild(div);
        });

        const submitDiv = document.createElement('div');
        submitDiv.className = 'md:col-span-2 text-center';
        submitDiv.innerHTML = `
            <button type="submit" class="bg-emerald-500 hover:bg-emerald-600 text-white font-bold py-3 px-8 rounded-full shadow-lg btn-glow green btn-shadow transform hover:scale-105 transition duration-300 ease-in-out mt-4 focus:ring-4 focus:ring-emerald-300 focus:outline-none">
                Detect Cancer
            </button>
        `;
        detectionForm.appendChild(submitDiv);
    }

    // Handle Detection form submission
    detectionForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        detectionResult.classList.add('hidden');
        detectionMessage.textContent = 'Detecting...';
        detectionMessage.classList.remove('error', 'success');

        const formData = {};
        detectionFeatures.forEach(feature => {
            const inputElement = document.getElementById(feature.name.toLowerCase().replace(/_/g, '-'));
            if (inputElement) {
                formData[feature.name] = inputElement.value;
            }
        });

        console.log('Data being sent to backend:', formData);

        try {
            const response = await fetch('/detect', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(formData)
            });
            const data = await response.json();

            if (response.ok) {
                detectionResult.classList.remove('hidden');
                resultMessage.textContent = `Prediction: ${data.message}`;
                resultProbability.textContent = `Probability of No Cancer: ${data.probability_no_cancer}, Probability of Cancer: ${data.probability_cancer}`;
                showMessage(detectionMessage, 'Detection successful!', 'success');

                // Enable download button
                const resultData = {
                    ...formData,
                    prediction: data.prediction,
                    probability_no_cancer: data.probability_no_cancer,
                    probability_cancer: data.probability_cancer,
                    message: data.message
                };

                const downloadBtn = document.getElementById("download-btn");
                if (downloadBtn) {
                    downloadBtn.classList.remove('hidden');
                    downloadBtn.onclick = () => {
                        downloadPdf(resultData);
                    };
                }

            } else {
                showMessage(detectionMessage, data.message || 'Error during detection.', 'error');
            }
        } catch (error) {
            console.error('Detection error:', error);
            showMessage(detectionMessage, 'An error occurred during detection. Please check your network connection and server logs.', 'error');
        }
    });

    // --- Model Comparison Section Logic ---

    /**
     * Fetches and displays model comparison data from the backend.
     */
    async function fetchModelComparisonData() {
        comparisonData.innerHTML = ''; // Clear previous content
        comparisonLoadingMessage.classList.remove('hidden'); // Show loading message
        comparisonMessage.textContent = ''; // Clear previous messages

        try {
            const response = await fetch('/model_comparison'); // Fetch data from new backend endpoint
            const data = await response.json();

            if (response.ok) {
                comparisonLoadingMessage.classList.add('hidden'); // Hide loading message

                let htmlContent = '<div class="overflow-x-auto"><table class="min-w-full bg-white rounded-lg shadow-md">';
                htmlContent += '<thead class="bg-gray-200 text-gray-700">';
                htmlContent += '<tr><th class="py-3 px-4 text-left">Model</th><th class="py-3 px-4 text-left">Accuracy</th><th class="py-3 px-4 text-left">Precision</th><th class="py-3 px-4 text-left">Recall</th><th class="py-3 px-4 text-left">F1-Score</th></tr>';
                htmlContent += '</thead><tbody>';

                let bestAccuracy = -1;
                let bestModelName = '';

                for (const modelName in data) {
                    const metrics = data[modelName];
                    htmlContent += `<tr class="border-b border-gray-200 hover:bg-gray-50">`;
                    htmlContent += `<td class="py-3 px-4 font-semibold">${modelName}</td>`;
                    htmlContent += `<td class="py-3 px-4">${metrics.accuracy}</td>`;
                    htmlContent += `<td class="py-3 px-4">${metrics.precision}</td>`;
                    htmlContent += `<td class="py-3 px-4">${metrics.recall}</td>`;
                    htmlContent += `<td class="py-3 px-4">${metrics.f1_score}</td>`;
                    htmlContent += `</tr>`;

                    if (metrics.accuracy > bestAccuracy) {
                        bestAccuracy = metrics.accuracy;
                        bestModelName = modelName;
                    }
                }
                htmlContent += '</tbody></table></div>';
                htmlContent += `<p class="text-lg font-bold text-center mt-6">Best Model: <span class="text-indigo-600">${bestModelName}</span> (Accuracy: ${bestAccuracy})</p>`;

                comparisonData.innerHTML = htmlContent;
                showMessage(comparisonMessage, 'Model comparison data loaded.', 'success');
            } else {
                comparisonLoadingMessage.classList.add('hidden');
                showMessage(comparisonMessage, data.message || 'Error fetching model comparison data.', 'error');
            }
        } catch (error) {
            console.error('Model comparison data fetch error:', error);
            comparisonLoadingMessage.classList.add('hidden');
            showMessage(comparisonMessage, 'An error occurred while fetching model comparison data. Please check your network connection and server logs.', 'error');
        }
    }

    // --- Analysis Section Logic ---
    let survivalStatusChartInstance = null;
    let genderDistributionChartInstance = null;
    let smokingHistoryChartInstance = null;
    let ageGroupChartInstance = null;
    let cancerByGenderChartInstance = null;

    function destroyCharts() {
        if (survivalStatusChartInstance) survivalStatusChartInstance.destroy();
        if (genderDistributionChartInstance) genderDistributionChartInstance.destroy();
        if (smokingHistoryChartInstance) smokingHistoryChartInstance.destroy();
        if (ageGroupChartInstance) ageGroupChartInstance.destroy();
        if (cancerByGenderChartInstance) cancerByGenderChartInstance.destroy();
    }

    async function fetchAnalysisData() {
        analysisMessage.textContent = 'Loading analysis data...';
        analysisMessage.classList.remove('error', 'success');

        try {
            const response = await fetch('/analysis');
            const data = await response.json();

            if (response.ok) {
                analysisMessage.textContent = ''; // Clear loading message
                updateAnalysisSummary(data);
                destroyCharts(); // Destroy existing charts before rendering new ones
                renderCharts(data);
                showMessage(analysisMessage, 'Analysis data loaded successfully!', 'success');
            } else {
                showMessage(analysisMessage, data.message || 'Error fetching analysis data.', 'error');
            }
        } catch (error) {
            console.error('Analysis data fetch error:', error);
            showMessage(analysisMessage, 'An error occurred while fetching analysis data. Please check your network connection and server logs.', 'error');
        }
    }

    function updateAnalysisSummary(data) {
        totalRecordsSpan.textContent = data.total_records;
        cancerCasesSummarySpan.textContent = data.survival_status_counts['1'] || 0;
        nonCancerCasesSummarySpan.textContent = data.survival_status_counts['0'] || 0;
        const cancerPercentage = (data.survival_status_counts['1'] / data.total_records * 100).toFixed(2);
        cancerPercentageSummarySpan.textContent = `${cancerPercentage}%`;
    }

    function renderCharts(data) {
        // Chart 1: Survival Status Distribution (Pie Chart)
        const survivalStatusCtx = document.getElementById('survivalStatusChart').getContext('2d');
        survivalStatusChartInstance = new Chart(survivalStatusCtx, {
            type: 'pie',
            data: {
                labels: ['No Cancer', 'Cancer'],
                datasets: [{
                    data: [data.survival_status_counts['0'] || 0, data.survival_status_counts['1'] || 0],
                    backgroundColor: ['#4299e1', '#ef4444'], // blue, red
                    hoverOffset: 4
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'top',
                    },
                    title: {
                        display: false,
                        text: 'Survival Status Distribution'
                    }
                }
            }
        });

        // Chart 2: Gender Distribution (Bar Chart)
        const genderDistributionCtx = document.getElementById('genderDistributionChart').getContext('2d');
        genderDistributionChartInstance = new Chart(genderDistributionCtx, {
            type: 'bar',
            data: {
                labels: Object.keys(data.gender_distribution),
                datasets: [{
                    label: 'Number of Individuals',
                    data: Object.values(data.gender_distribution),
                    backgroundColor: ['#63b3ed', '#a78bfa', '#f6ad55'], // light blue, light purple, orange
                    borderColor: ['#4299e1', '#8b5cf6', '#ed8936'],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        display: false,
                    },
                    title: {
                        display: false,
                        text: 'Gender Distribution'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });

        // Chart 3: Smoking History Distribution (Pie Chart)
        const smokingHistoryCtx = document.getElementById('smokingHistoryChart').getContext('2d');
        smokingHistoryChartInstance = new Chart(smokingHistoryCtx, {
            type: 'pie',
            data: {
                labels: ['No Smoking History (0)', 'Smoking History (1)'],
                datasets: [{
                    data: [data.smoking_history_distribution['0'] || 0, data.smoking_history_distribution['1'] || 0],
                    backgroundColor: ['#48bb78', '#f6e05e'], // green, yellow
                    hoverOffset: 4
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'top',
                    },
                    title: {
                        display: false,
                        text: 'Smoking History Distribution'
                    }
                }
            }
        });

        // Chart 4: Age Group Distribution (Bar Chart)
        const ageGroupCtx = document.getElementById('ageGroupChart').getContext('2d');
        ageGroupChartInstance = new Chart(ageGroupCtx, {
            type: 'bar',
            data: {
                labels: Object.keys(data.age_group_distribution),
                datasets: [{
                    label: 'Number of Individuals',
                    data: Object.values(data.age_group_distribution),
                    backgroundColor: '#81e6d9', // teal
                    borderColor: '#38b2ac',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        display: false,
                    },
                    title: {
                        display: false,
                        text: 'Age Group Distribution'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });

        // Chart 5: Cancer Cases by Gender (Bar Chart)
        const cancerByGenderCtx = document.getElementById('cancerByGenderChart').getContext('2d');
        cancerByGenderChartInstance = new Chart(cancerByGenderCtx, {
            type: 'bar',
            data: {
                labels: Object.keys(data.cancer_cases_by_gender),
                datasets: [{
                    label: 'Cancer Cases',
                    data: Object.values(data.cancer_cases_by_gender),
                    backgroundColor: ['#f56565', '#ed8936'], // red, orange
                    borderColor: ['#c53030', '#dd6b20'],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        display: false,
                    },
                    title: {
                        display: false,
                        text: 'Cancer Cases by Gender'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }

function downloadPdf(resultData) {
    fetch('/download_result', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(resultData)
    })
    .then(response => {
        if (response.ok) return response.blob();
        else throw new Error("PDF download failed");
    })
    .then(blob => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = "detection_result.pdf";
        document.body.appendChild(a);
        a.click();
        a.remove();
    })
    .catch(error => console.error("PDF error:", error));
}


    // --- Initial Setup on Page Load ---
    // Show the home section by default
    showSection(homeSection);
    // Update navigation button visibility based on initial login state (initially logged out)
    updateNavVisibility();
});
