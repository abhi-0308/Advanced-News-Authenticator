document.addEventListener('DOMContentLoaded', function() {
  const analyzeBtn = document.getElementById('analyzeBtn');
  const newsInput = document.getElementById('newsInput');
  const resultContainer = document.getElementById('resultContainer');
  const confidenceBar = document.getElementById('confidenceBar');
  const confidenceValue = document.getElementById('confidenceValue');
  const resultIcon = document.getElementById('resultIcon');
  const resultText = document.getElementById('resultText');
  const redFlagsList = document.getElementById('redFlagsList');

  const RED_FLAGS = {
    'flat_earth_claim': {
      patterns: [/(earth|world|planet).*(flat|disc|disk)|(flat|disc|disk).*(earth|world|planet)/i],
      description: 'Debunked flat Earth theory',
      severity: 'critical'
    },
    'false_medical_claim': {
      patterns: [/(autism|adhd|disorders).*(caused by|from).*(sugar|food|diet)/i,
                 /(vaccine|vaccination).*(autism|tracking|mind control)/i],
      description: 'False medical claim',
      severity: 'critical'
    },
    'false_economic_claim': {
      patterns: [/(poverty|hunger|world hunger).*(declining|decreasing|reducing|solved)/i],
      description: 'Misleading economic claim',
      severity: 'critical'
    },
    'climate_denial_claim': {
      patterns: [/(climate change|global warming).*(hoax|fake|not real)/i],
      description: 'Climate science denial',
      severity: 'critical'
    },
    'vaccine_misinformation': {
      patterns: [/(vaccine|vaccination).*(autism|tracking|mind control)/i],
      description: 'Vaccine misinformation',
      severity: 'critical'
    },
    'covid_conspiracy': {
      patterns: [/(covid|coronavirus).*(fake|hoax|planned|conspiracy)/i],
      description: 'COVID-19 conspiracy theory',
      severity: 'critical'
    },
    'fake_authority_claim': {
      patterns: [/(scientist|expert|research|study).*(confirm|prove|show|demonstrate|reveal|find)/i, 
                 /(confirm|prove|show|demonstrate|reveal|find).*(scientist|expert|research|study)/i],
      description: 'Misleading authority appeal',
      severity: 'standard'
    },
    'exaggerated_claim': {
      patterns: [/(groundbreaking|revolutionary|shocking|stunning|amazing).*(study|research|discovery|finding|result)/i,
                 /(study|research|discovery|finding|result).*(groundbreaking|revolutionary|shocking|stunning|amazing)/i],
      description: 'Overstated scientific claim',
      severity: 'standard'
    },
    'conspiracy_claim': {
      patterns: [/(conspiracy|secret|cover-up|hide|suppress).*(government|truth|fact)/i],
      description: 'Unverified conspiracy claim',
      severity: 'standard'
    },
    'miracle_claim': {
      patterns: [/(miracle|magical|incredible).*(cure|solution|remedy|treatment)/i],
      description: 'Unrealistic miracle solution',
      severity: 'standard'
    }
  };

  function detectRedFlags(text, serverDetectedPatterns = []) {
    redFlagsList.innerHTML = '';
    const lowerText = text.toLowerCase();
    let flagsFound = 0;

    if (serverDetectedPatterns && serverDetectedPatterns.length > 0) {
      serverDetectedPatterns.forEach(pattern => {
        if (RED_FLAGS[pattern]) {
          addRedFlag(RED_FLAGS[pattern].description, true, RED_FLAGS[pattern].severity);
          flagsFound++;
        } else {
          addRedFlag(`Suspicious pattern: ${pattern.replace(/_/g, ' ')}`, true, 'standard');
          flagsFound++;
        }
      });
    }

    for (const [key, flagData] of Object.entries(RED_FLAGS)) {
      if (serverDetectedPatterns && serverDetectedPatterns.includes(key)) continue;
      
      for (const pattern of flagData.patterns) {
        if (pattern.test(lowerText)) {
          addRedFlag(flagData.description, true, flagData.severity);
          flagsFound++;
          break; 
        }
      }
    }

    if (/\b100%|\bguaranteed\b|\balways\b|\bnever\b|\bdefinitely\b/i.test(lowerText) && 
        !serverDetectedPatterns.includes('absolute_claim')) {
      addRedFlag('Absolute claims without nuance', true, 'standard');
      flagsFound++;
    }

    if (/\bthey don't want you to know\b|\bwhat they aren't telling you\b|\bsecret they're hiding\b/i.test(lowerText) && 
        !serverDetectedPatterns.includes('exclusive_knowledge')) {
      addRedFlag('Appeal to exclusive knowledge', true, 'standard');
      flagsFound++;
    }

    if (flagsFound === 0) {
      addRedFlag('No obvious red flags detected', false);
    }
  }

  function addRedFlag(message, isWarning = true, severity = 'standard') {
    const li = document.createElement('li');
    li.textContent = message;
    
    if (!isWarning) {
      li.style.color = '#4CAF50'; 
    } else if (severity === 'critical') {
      li.style.color = '#F44336'; 
      li.style.fontWeight = 'bold';
    } else {
      li.style.color = '#FF9800'; 
    }
    
    redFlagsList.appendChild(li);
  }
  
  analyzeBtn.addEventListener('click', analyzeNews);

  async function analyzeNews() {
    const text = newsInput.value.trim();
    
    if (!text) {
      alert('Please enter a news article to analyze');
      return;
    }

    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
    resultContainer.classList.remove('visible');
    resultContainer.classList.add('hidden');

    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify({ text: text })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      displayResults(data, text);
    } catch (error) {
      console.error('Error:', error);
      showError(error.message);
    } finally {
      analyzeBtn.disabled = false;
      analyzeBtn.innerHTML = '<i class="fas fa-search"></i> Analyze';
      resultContainer.classList.remove('hidden');
      resultContainer.classList.add('visible');
    }
  }

  function displayResults(data, originalText) {
    const confidencePercent = Math.round(data.confidence * 100);
    confidenceBar.style.width = `${confidencePercent}%`;
    confidenceValue.textContent = `${confidencePercent}%`;

    if (data.is_fake) {
      resultIcon.className = 'fas fa-times-circle';
      resultIcon.style.color = 'var(--fake-color)';
      resultText.textContent = 'This appears to be FAKE NEWS';
      confidenceBar.style.backgroundColor = 'var(--fake-color)';
    } else {
      resultIcon.className = 'fas fa-check-circle';
      resultIcon.style.color = 'var(--real-color)';
      resultText.textContent = 'This appears to be REAL NEWS';
      confidenceBar.style.backgroundColor = 'var(--real-color)';
    }

    detectRedFlags(originalText, data.detected_patterns || []);

    resultContainer.scrollIntoView({ behavior: 'smooth' });
  }

  function showError(message) {
    resultIcon.className = 'fas fa-exclamation-triangle';
    resultIcon.style.color = 'var(--neutral-color)';
    resultText.textContent = `Error: ${message || 'Analysis failed'}`;
    confidenceBar.style.width = '0%';
    confidenceValue.textContent = '0%';
    redFlagsList.innerHTML = '<li style="color: var(--neutral-color)">Unable to analyze content</li>';
  }

  newsInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && e.ctrlKey) {
      e.preventDefault();
      analyzeNews();
    }
  });
});