const API_URL = "http://127.0.0.1:5000/api/predict";

let churnChart = null;

// Get HTML elements
const form = document.querySelector("form");
const predictBtn = document.getElementById("predictBtn");
const resultBox = document.getElementById("result");
const predictionText = document.getElementById("predictionText");
const probabilityText = document.getElementById("probabilityText");
const resultMessage = document.getElementById("resultMessage");

// Handle prediction form
form.addEventListener("submit", async function (event) {
    event.preventDefault();

    predictBtn.disabled = true;
    predictBtn.textContent = "Predicting...";

    try {
        const formData = new FormData(form);
        const customerData = {};

        // Collect all 19 customer details
        for (const [key, value] of formData.entries()) {
            customerData[key] = value;
        }

        // Convert numeric fields into numbers
        const numericFields = [
            "SeniorCitizen",
            "tenure",
            "MonthlyCharges",
            "TotalCharges"
        ];

        numericFields.forEach(field => {
            customerData[field] = Number(customerData[field]);
        });

        // Send customer data to Flask backend
        const response = await fetch(API_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(customerData)
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Prediction failed.");
        }

        // Display prediction
        const prediction = data.prediction;
        let probability = Number(data.churn_probability);

        // Convert probability to percentage if needed
        if (probability <= 1) {
            probability *= 100;
        }

        predictionText.textContent =
            prediction === 1 || prediction === "Yes"
                ? "High Churn Risk"
                : "Low Churn Risk";

        probabilityText.textContent =
    "Churn Probability: " + probability.toFixed(2) + "%";

const retentionProbability = 100 - probability;

resultMessage.textContent =
    "Retention Probability: " +
    retentionProbability.toFixed(2) + "%";

        resultMessage.textContent =
            data.message || "Prediction completed successfully.";

        resultBox.style.display = "block";

        // Draw prediction chart
        const canvas = document.querySelector("canvas");

        if (canvas && typeof Chart !== "undefined") {
            if (churnChart) {
                churnChart.destroy();
            }

            churnChart = new Chart(canvas, {
                type: "doughnut",
                data: {
                    labels: ["Churn Probability", "Retention Probability"],
                    datasets: [{
                        data: [probability, 100 - probability]
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: {
                            position: "bottom"
                        }
                    }
                }
            });
        }

    } catch (error) {
        alert("Error: " + error.message);
        console.error(error);

    } finally {
        predictBtn.disabled = false;
        predictBtn.textContent = "Predict Churn";
    }
});