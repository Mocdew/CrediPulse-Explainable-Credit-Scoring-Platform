// CrediPulse Interactive Frontend Application Logic
let schema = null;
let presets = null;
let currentThreshold = 0.17;
let shapChart = null;
let batchDataCsv = null;

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', async () => {
    initTabs();
    await loadSchemaAndPresets();
    initChart();
    // Trigger initial prediction with default form values
    evaluateCurrentForm();
});

// Tab Navigation
function initTabs() {
    const tabs = document.querySelectorAll('.nav-tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

            tab.classList.add('active');
            const targetId = tab.getAttribute('data-tab');
            document.getElementById(targetId).classList.add('active');
        });
    });
}

// Load Schema & Presets from Backend API
async function loadSchemaAndPresets() {
    try {
        const [schemaRes, presetsRes] = await Promise.all([
            fetch('/api/schema'),
            fetch('/api/presets')
        ]);
        schema = await schemaRes.json();
        presets = await presetsRes.json();

        populateDropdowns();
    } catch (err) {
        console.error('Error loading schema/presets:', err);
    }
}

// Populate Categorical Dropdowns
function populateDropdowns() {
    if (!schema || !schema.categories) return;

    for (const [field, options] of Object.entries(schema.categories)) {
        const select = document.getElementById(field);
        if (select) {
            select.innerHTML = '';
            options.forEach(opt => {
                const el = document.createElement('option');
                el.value = opt;
                el.textContent = opt;
                select.appendChild(el);
            });
        }
    }
}

// Load Preset Applicant
function loadPreset(presetKey) {
    if (!presets || !presets[presetKey]) return;
    const data = presets[presetKey].data;

    for (const [field, val] of Object.entries(data)) {
        const el = document.getElementById(field);
        if (el) {
            el.value = val;
        }
    }

    // Auto trigger evaluation
    evaluateCurrentForm();
}

// Reset form
function resetForm() {
    document.getElementById('creditForm').reset();
    populateDropdowns();
    evaluateCurrentForm();
}

// Read current form values into JS object
function getFormData() {
    const form = document.getElementById('creditForm');
    const formData = new FormData(form);
    const applicant = {};
    for (const [key, value] of formData.entries()) {
        applicant[key] = value;
    }
    return applicant;
}

// Handle Form Submission
async function handleSingleSubmit(e) {
    if (e) e.preventDefault();
    await evaluateCurrentForm();
}

// Evaluate Form via /api/predict
async function evaluateCurrentForm() {
    const applicant = getFormData();
    const btn = document.getElementById('evaluateBtn');
    if (btn) btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Evaluating...';

    try {
        const res = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                applicant: applicant,
                threshold: currentThreshold
            })
        });
        const result = await res.json();
        if (result.success) {
            renderResults(result.data);
        } else {
            alert('Prediction error: ' + result.error);
        }
    } catch (err) {
        console.error('Inference error:', err);
    } finally {
        if (btn) btn.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles"></i> Evaluate Credit Risk';
    }
}

// Render Prediction Dashboard
function renderResults(data) {
    // 1. Decision Badge & Risk Tier
    const badge = document.getElementById('decisionBadge');
    const decisionText = document.getElementById('decisionText');
    const tierTag = document.getElementById('riskTierTag');

    badge.className = 'decision-badge ' + data.status_code;
    badge.innerHTML = data.status_code === 'approved' 
        ? '<i class="fa-solid fa-circle-check"></i> ' + data.decision
        : '<i class="fa-solid fa-triangle-exclamation"></i> ' + data.decision;

    tierTag.textContent = data.risk_tier;
    tierTag.style.borderColor = data.tier_color;
    tierTag.style.color = data.tier_color;

    // 2. Default Probability & Progress bar
    document.getElementById('probBadText').textContent = data.prob_bad_pct + '%';
    const progressBar = document.getElementById('riskProgressBar');
    progressBar.style.width = Math.min(Math.max(data.prob_bad_pct, 4), 100) + '%';
    document.getElementById('thresholdComparison').textContent = `Threshold Cutoff: ${(data.threshold_used * 100).toFixed(0)}%`;

    // 3. Credit Scorecard Points
    document.getElementById('scoreText').textContent = data.credit_score;
    // Score dot position: map [300, 850] -> [0%, 100%]
    const scorePct = Math.min(Math.max(((data.credit_score - 300) / (850 - 300)) * 100, 0), 100);
    document.getElementById('scoreDot').style.left = scorePct + '%';

    // 4. Adverse Reason Codes
    const incList = document.getElementById('riskIncreasingList');
    incList.innerHTML = '';
    if (data.adverse_reasons && data.adverse_reasons.length > 0) {
        data.adverse_reasons.forEach(r => {
            const item = document.createElement('div');
            item.className = 'reason-item risk-increasing';
            item.innerHTML = `
                <div class="reason-item-header">
                    <span>${r.label}</span>
                    <span class="impact-badge pos">+${r.impact}</span>
                </div>
                <div class="reason-val">Value: <strong>${r.value}</strong></div>
            `;
            incList.appendChild(item);
        });
    } else {
        incList.innerHTML = '<div class="placeholder-text">No significant elevated risk drivers found.</div>';
    }

    // 5. Positive / Mitigating Factors
    const decList = document.getElementById('riskDecreasingList');
    decList.innerHTML = '';
    if (data.positive_factors && data.positive_factors.length > 0) {
        data.positive_factors.forEach(r => {
            const item = document.createElement('div');
            item.className = 'reason-item risk-decreasing';
            item.innerHTML = `
                <div class="reason-item-header">
                    <span>${r.label}</span>
                    <span class="impact-badge neg">${r.impact}</span>
                </div>
                <div class="reason-val">Value: <strong>${r.value}</strong></div>
            `;
            decList.appendChild(item);
        });
    } else {
        decList.innerHTML = '<div class="placeholder-text">No mitigating factors noted.</div>';
    }

    // 6. Update SHAP chart
    updateShapChart(data.chart_features);
}

// Decision Threshold Handlers
function setThreshold(val) {
    currentThreshold = parseFloat(val);
    document.getElementById('thresholdSlider').value = val;
    document.getElementById('thresholdValText').textContent = `${val} (${(val * 100).toFixed(0)}%)`;
    
    document.querySelectorAll('.chip-btn').forEach(btn => {
        if (parseFloat(btn.textContent.match(/[0-9.]+/)[0]) === val) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });

    evaluateCurrentForm();
}

function onThresholdChange(val) {
    currentThreshold = parseFloat(val);
    document.getElementById('thresholdValText').textContent = `${val} (${(val * 100).toFixed(0)}%)`;
    evaluateCurrentForm();
}

// Chart.js SHAP Bar Chart
function initChart() {
    const ctx = document.getElementById('shapChart').getContext('2d');
    shapChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'SHAP Feature Contribution',
                data: [],
                backgroundColor: [],
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const val = context.raw;
                            return val > 0 ? ` +${val} (Elevates Default Risk)` : ` ${val} (Reduces Risk / Safe)`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: '#2e3d5b' },
                    ticks: { color: '#94a3b8' }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#f3f4f6', font: { size: 11 } }
                }
            }
        }
    });
}

function updateShapChart(features) {
    if (!shapChart || !features) return;
    
    // Sort and take top 10 impactful features
    const topFeats = [...features].sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact)).slice(0, 10).reverse();

    shapChart.data.labels = topFeats.map(f => f.label);
    shapChart.data.datasets[0].data = topFeats.map(f => f.impact);
    shapChart.data.datasets[0].backgroundColor = topFeats.map(f => f.impact > 0 ? '#f43f5e' : '#10b981');
    shapChart.update();
}

// ================= BATCH PROCESSING =================
function handleFileSelected(input) {
    if (input.files && input.files[0]) {
        const file = input.files[0];
        document.getElementById('selectedFileName').textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
        document.getElementById('selectedFileInfo').style.display = 'flex';
    }
}

async function submitBatchUpload() {
    const fileInput = document.getElementById('csvFileInput');
    if (!fileInput.files || !fileInput.files[0]) return;

    const btn = document.getElementById('runBatchBtn');
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing...';

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('threshold', currentThreshold);

    try {
        const res = await fetch('/api/predict-batch', {
            method: 'POST',
            body: formData
        });
        const data = await res.json();
        if (data.success) {
            renderBatchResults(data);
        } else {
            alert('Batch processing error: ' + data.error);
        }
    } catch (err) {
        console.error('Batch error:', err);
    } finally {
        btn.innerHTML = '<i class="fa-solid fa-bolt"></i> Run Batch Scoring';
    }
}

function renderBatchResults(data) {
    batchDataCsv = data.csv_data;
    document.getElementById('batchResultsSection').style.display = 'block';

    // Summary KPIs
    document.getElementById('batchTotalCount').textContent = data.summary.total_applicants;
    document.getElementById('batchApprovedCount').textContent = data.summary.approved_count;
    document.getElementById('batchApprovedRate').textContent = `${data.summary.approval_rate_pct}% approval rate`;
    document.getElementById('batchDeclinedCount').textContent = data.summary.declined_count;
    document.getElementById('batchAvgRisk').textContent = `${data.summary.avg_default_risk_pct}%`;

    // Preview Table
    const thead = document.getElementById('batchTableHead');
    const tbody = document.getElementById('batchTableBody');
    thead.innerHTML = '';
    tbody.innerHTML = '';

    const priorityCols = ['Prediction', 'Default_Risk_Prob', 'Credit_Score', 'checking_status', 'duration_months', 'credit_amount', 'purpose', 'age'];
    
    // Header
    const trHead = document.createElement('tr');
    priorityCols.forEach(col => {
        const th = document.createElement('th');
        th.textContent = col.replace('_', ' ');
        trHead.appendChild(th);
    });
    thead.appendChild(trHead);

    // Rows
    data.preview.forEach(row => {
        const tr = document.createElement('tr');
        priorityCols.forEach(col => {
            const td = document.createElement('td');
            if (col === 'Prediction') {
                const statusClass = row[col] === 'APPROVED' ? 'approved' : 'declined';
                td.innerHTML = `<span class="status-pill ${statusClass}">${row[col]}</span>`;
            } else if (col === 'Default_Risk_Prob') {
                td.textContent = (row[col] * 100).toFixed(1) + '%';
            } else {
                td.textContent = row[col] !== undefined ? row[col] : '';
            }
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    });
}

function exportBatchCsv() {
    if (!batchDataCsv) return;
    const blob = new Blob([batchDataCsv], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `scored_applicants_threshold_${currentThreshold}.csv`;
    link.click();
}

// Download Sample CSV
function downloadSampleCsv(e) {
    if (e) e.preventDefault();
    const sampleCsv = `checking_status,duration_months,credit_history,purpose,credit_amount,savings_status,employment_since,installment_rate,personal_status_sex,other_debtors,residence_since,property,age,other_installment_plans,housing,existing_credits,job,num_dependents,telephone,foreign_worker
< 0 DM,6,critical account / other credits elsewhere,radio/television,1169,unknown/no savings account,>= 7 years,4,male: single,none,4,real estate,67,none,own,2,skilled employee/official,1,yes, registered,yes
0-200 DM,48,existing credits paid duly till now,radio/television,5951,< 100 DM,1-4 years,2,female: divorced/separated/married,none,2,real estate,22,none,own,1,skilled employee/official,1,none,yes
no checking account,12,critical account / other credits elsewhere,education,2096,< 100 DM,4-7 years,2,male: single,none,3,real estate,49,none,own,1,unskilled resident,2,none,yes
< 0 DM,42,existing credits paid duly till now,furniture/equipment,7882,< 100 DM,4-7 years,2,male: single,guarantor,4,building society savings/life insurance,45,none,for free,1,skilled employee/official,2,none,yes
< 0 DM,24,delay in paying off in the past,car (new),4870,< 100 DM,1-4 years,3,male: single,none,4,unknown/no property,53,none,for free,2,skilled employee/official,2,none,yes
>= 200 DM,18,existing credits paid duly till now,business,2238,500-1000 DM,1-4 years,2,female: single,none,1,car or other property,32,none,own,1,skilled employee/official,1,none,yes
no checking account,24,all credits at this bank paid duly,car (used),3000,>= 1000 DM,>= 7 years,3,male: married/widowed,co-applicant,4,real estate,55,bank,own,3,management/self-employed/highly qualified,1,yes, registered,yes`;

    const blob = new Blob([sampleCsv], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'sample_german_credit_applicants.csv';
    link.click();
}
