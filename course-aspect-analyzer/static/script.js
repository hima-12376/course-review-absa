document.addEventListener('DOMContentLoaded', () => {

    const analyzeBtn = document.getElementById('analyzeBtn');
    const textarea = document.getElementById('feedbackInput');
    const grid = document.getElementById('insightsGrid');
    let macroChart = null;
    let aspectChart = null;

    let sentimentChart = null;

    // 🔥 Generate Cards from API Data
    const generateCards = (data) => {
        grid.innerHTML = '';

        data.forEach((item, index) => {

            const card = document.createElement('div');
            card.className = 'card glass-panel';
            card.style.animationDelay = `${index * 0.1}s`;

            let colorVar, glowVar;

            if (item.status === 'positive') {
                colorVar = 'var(--accent-green)';
                glowVar = 'rgba(16, 185, 129, 0.2)';
            } else if (item.status === 'negative') {
                colorVar = 'var(--accent-red)';
                glowVar = 'rgba(239, 68, 68, 0.2)';
            } else {
                colorVar = 'var(--accent-purple)';
                glowVar = 'rgba(159, 122, 234, 0.2)';
            }

            card.style.setProperty('--accent-color', colorVar);
            card.style.setProperty('--accent-glow', glowVar);

            card.innerHTML = `
                <div class="card-header">
                    <span class="aspect-name">${item.aspect}</span>
                    <span class="sentiment-badge">${item.sentiment}%</span>
                </div>
                <div class="bar-container">
                    <div class="bar-fill" style="--fill-percentage: ${item.sentiment}%"></div>
                </div>
                <p class="card-desc">${item.description}</p>
            `;

            grid.appendChild(card);
        });
    };

    // 🔥 Render Chart
    const renderChart = (data) => {

        const ctx = document.getElementById('sentimentChart').getContext('2d');

        const labels = data.map(item => item.aspect);
        const sentiments = data.map(item => item.sentiment);

        const bgColors = data.map(item => {
            if (item.status === 'positive') return 'rgba(16, 185, 129, 0.6)';
            if (item.status === 'negative') return 'rgba(239, 68, 68, 0.6)';
            return 'rgba(159, 122, 234, 0.6)';
        });

        const borderColors = data.map(item => {
            if (item.status === 'positive') return 'rgba(16, 185, 129, 1)';
            if (item.status === 'negative') return 'rgba(239, 68, 68, 1)';
            return 'rgba(159, 122, 234, 1)';
        });

        if (sentimentChart) {
            sentimentChart.data.labels = labels;
            sentimentChart.data.datasets[0].data = sentiments;
            sentimentChart.data.datasets[0].backgroundColor = bgColors;
            sentimentChart.data.datasets[0].borderColor = borderColors;
            sentimentChart.update();
        } else {
            sentimentChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Sentiment Confidence (%)',
                        data: sentiments,
                        backgroundColor: bgColors,
                        borderColor: borderColors,
                        borderWidth: 1,
                        borderRadius: 6
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100
                        }
                    }
                }
            });
        }
    };

    function renderAdvancedCharts(data) {

    if (!data || data.length === 0) return;

    let positiveScore = 0;
    let negativeScore = 0;
    let neutralScore = 0;
    let totalConfidence = 0;

    data.forEach(item => {

        totalConfidence += item.sentiment;

        if (item.status === "positive") {
            positiveScore += item.sentiment;
        } 
        else if (item.status === "negative") {
            negativeScore += item.sentiment;
        } 
        else {
            neutralScore += item.sentiment;
        }
    });

    const total = positiveScore + negativeScore + neutralScore || 1;

    const macroData = [
        Math.round((positiveScore / total) * 100),
        Math.round((neutralScore / total) * 100),
        Math.round((negativeScore / total) * 100)
    ];

    // 🔵 Overall Sentiment Ring
    if (macroChart) macroChart.destroy();
    macroChart = new Chart(document.getElementById("macroChart"), {
        type: "doughnut",
        data: {
            labels: ["Positive", "Neutral", "Negative"],
            datasets: [{
                data: macroData,
                backgroundColor: ["#00f2c3", "#7000ff", "#ff5d8f"],
                borderWidth: 0
            }]
        },
        options: {
            cutout: "75%",
            plugins: { legend: { position: "bottom" } }
        }
    });

    // 🍩 Aspect Drilldown → Show weighted breakdown of ALL aspects
    if (aspectChart) aspectChart.destroy();
    aspectChart = new Chart(document.getElementById("aspectChart"), {
        type: "doughnut",
        data: {
            labels: data.map(d => d.aspect),
            datasets: [{
                data: data.map(d => d.sentiment),
                backgroundColor: [
                    "#00f2c3",
                    "#7000ff",
                    "#ff5d8f",
                    "#00bcd4",
                    "#ff9800"
                ],
                borderWidth: 0
            }]
        },
        options: {
            cutout: "65%",
            plugins: { legend: { position: "bottom" } }
        }
    });

    // 🟣 Confidence = Average confidence
    const avgConfidence = Math.round(totalConfidence / data.length);

    const circle = document.getElementById("confidenceCircle");
    const circumference = 471;
    const offset = circumference - (avgConfidence / 100) * circumference;

    circle.style.strokeDashoffset = offset;
    document.getElementById("confidenceText").innerText = avgConfidence + "%";
}
    // 🚀 Connect to Backend
    analyzeBtn.addEventListener('click', async () => {

        const text = textarea.value.trim();

        if (!text) {
            alert("Please enter a review first.");
            return;
        }

        analyzeBtn.innerText = "Processing...";
        analyzeBtn.disabled = true;

        try {
            const response = await fetch("http://127.0.0.1:5000/analyze", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ review: text })
            });

            if (!response.ok) {
                throw new Error("Server error");
            }

            const data = await response.json();

            if (data.error) {
                alert(data.error);
                return;
            }

            if (data.length === 0) {
                alert("No aspects detected.");
                return;
            }

            generateCards(data);
            renderChart(data);
            renderAdvancedCharts(data);
        } catch (error) {
            console.error(error);
            alert("Cannot connect to backend. Make sure Flask server is running.");
        }

        analyzeBtn.innerText = "Analyze Feedback";
        analyzeBtn.disabled = false;
    });

});