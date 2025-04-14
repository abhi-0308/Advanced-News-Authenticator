import torch
import pickle
import re
from pathlib import Path
from tensorflow.keras.preprocessing.sequence import pad_sequences
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_VOCAB = 15000 
MAX_LEN = 256
MODEL_DIR = Path('aiproject/model')

class LiteFakeNewsDetector(torch.nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, 96)
        self.lstm = torch.nn.LSTM(96, 64,
                               num_layers=1,
                               bidirectional=True,
                               batch_first=True)
        self.fc = torch.nn.Linear(128, 1)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        return self.sigmoid(self.fc(x[:, -1, :]))

class NewsClassifier:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        self.model.eval()
    
    def _load_tokenizer(self):
        try:
            with open(MODEL_DIR / 'tokenizer.pkl', 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Tokenizer load failed: {e}")
            raise
    
    def _load_model(self):
        model = LiteFakeNewsDetector(MAX_VOCAB).to(self.device)
        try:
            model.load_state_dict(torch.load(MODEL_DIR / 'optimized_model.pt',
                                         map_location=self.device))
            return model
        except Exception as e:
            logger.error(f"Model load failed: {e}")
            raise
    
    @staticmethod
    def clean_text(text):
        """Enhanced cleaning with better fake news detection"""
        if not isinstance(text, str):
            return "", [], ""

        text = text.lower()
        
        matches = []
        
        fake_patterns = {
            r'\b(earth|world|planet).*(flat|disc|disk)\b': 'flat_earth_claim',
            r'\b(flat|disc|disk).*(earth|world|planet)\b': 'flat_earth_claim',
            r'\b(autism|adhd|disorders).*(caused by|from).*(sugar|food|diet)\b': 'false_medical_claim',
            r'\b(sugar|food|diet).*(cause|causes).*(autism|adhd|disorders)\b': 'false_medical_claim',
            r'\b(poverty|hunger|world hunger).*(declining|decreasing|reducing|solved)\b': 'false_economic_claim',
            r'\b(climate change|global warming).*(hoax|fake|not real|created|made up|invented)\b': 'climate_denial_claim',
            r'\b(scientist).*(create|creating|created|invent|inventing|invented).*(climate change|global warming)\b': 'climate_denial_claim',
            r'\b(climate change|global warming).*(funding|money|profit|research grant)\b': 'climate_funding_conspiracy',
            r'\b(vaccine|vaccination).*(autism|tracking|mind control)\b': 'vaccine_misinformation',
            r'\b(covid|coronavirus).*(fake|hoax|planned|conspiracy)\b': 'covid_conspiracy',
            r'\b(scientist|expert|research|study).*(confirm|prove|show|demonstrate|reveal|find)\b': 'fake_authority_claim',
            r'\b(confirm|prove|show|demonstrate|reveal|find).*(scientist|expert|research|study)\b': 'fake_authority_claim',
            r'\b(groundbreaking|revolutionary|shocking|stunning|amazing).*(study|research|discovery|finding|result)\b': 'exaggerated_claim',
            r'\b(study|research|discovery|finding|result).*(groundbreaking|revolutionary|shocking|stunning|amazing)\b': 'exaggerated_claim',
            r'\b(conspiracy|secret|cover-up|hide|suppress).*(government|truth|fact)\b': 'conspiracy_claim',
            r'\b(miracle|magical|incredible|amazing).*(cure|solution|remedy|treatment|heals)\b': 'miracle_claim',
            r'\b(cure|solution|remedy|treatment|heals).*(miracle|magical|incredible|amazing)\b': 'miracle_claim',
            r'\b(cure|cures|heal|heals).*(cancer|disease|illness).*(days|weeks|overnight)\b': 'miracle_cure_timeframe',
            r'\bcure.*(no side effects|without side effects)\b': 'impossible_medical_claim',
            r'\b(instantly|immediately|quickly|within days|in days|in a week|in weeks).*(cure|heal|fix|solve).*(cancer|disease|illness)\b': 'quick_cure_claim',
            r'\b(cancer|disease).*(cure|cured|cures|healing|healed).*(herb|natural|plant|tea|extract)\b': 'miracle_natural_cure',
            r'\b(they don\'t want you to know|what they aren\'t telling you)\b': 'exclusive_knowledge',
            r'\b(100%|guaranteed|always|never|definitely)\b': 'absolute_claim'
        }
        
        fake_score = 0
        original_text = text  # Store original for later pattern matching
        
        for pattern, replacement in fake_patterns.items():
            if re.search(pattern, text):
                matches.append(replacement)
                fake_score += 1
                text = re.sub(pattern, replacement, text)
        
        if matches:
            logger.info(f"Fake news patterns detected: {matches}")
        
        text = re.sub(r'[^a-z\s.,!?]', '', text)
        cleaned_text = re.sub(r'\s+', ' ', text).strip()
        
        return cleaned_text, matches, original_text

    def predict(self, text):
        try:
            cleaned, matches, original_text = self.clean_text(text)
            
            manual_fake_score = 0
            
            critical_fake_claims = {
                'flat_earth_claim': 0.35,
                'false_medical_claim': 0.30,
                'climate_denial_claim': 0.35,  # Increased weight
                'climate_funding_conspiracy': 0.35,  # New pattern with high weight
                'vaccine_misinformation': 0.35,
                'covid_conspiracy': 0.35,
                'false_economic_claim': 0.30,
                'miracle_claim': 0.30,  # Moved from standard_fake_claims
                'miracle_cure_timeframe': 0.40,  # New pattern with high weight
                'impossible_medical_claim': 0.35,  # New pattern
                'quick_cure_claim': 0.40,  # New pattern with high weight
                'miracle_natural_cure': 0.35  # New pattern
            }
            
            standard_fake_claims = {
                'fake_authority_claim': 0.15,
                'exaggerated_claim': 0.10,
                'conspiracy_claim': 0.20,
                'exclusive_knowledge': 0.15,
                'absolute_claim': 0.10
            }
            
            for match in matches:
                if match in critical_fake_claims:
                    manual_fake_score += critical_fake_claims[match]
                    logger.info(f"Critical fake claim detected: {match}")
                elif match in standard_fake_claims:
                    manual_fake_score += standard_fake_claims[match]
                    logger.info(f"Standard fake claim detected: {match}")
            
            if re.search(r'flat earth|earth is flat|planet is flat|world is flat', original_text.lower()):
                manual_fake_score += 0.35
                if 'flat_earth_claim' not in matches:
                    matches.append('flat_earth_claim')
                logger.info("Manual detection: Flat earth claim detected")
                
            if re.search(r'autism.*(sugar|diet|food|cause)', original_text.lower()) or \
               re.search(r'(sugar|diet|food).*(cause|causes).*autism', original_text.lower()):
                manual_fake_score += 0.30
                if 'false_medical_claim' not in matches:
                    matches.append('false_medical_claim')
                logger.info("Manual detection: False autism claim detected")
                
            if re.search(r'(poverty|hunger).*(solved|eliminated|fixed|declining)', original_text.lower()):
                manual_fake_score += 0.30
                if 'false_economic_claim' not in matches:
                    matches.append('false_economic_claim')
                logger.info("Manual detection: False economic claim detected")
                
            if re.search(r'vaccine.*(cause|causes).*autism', original_text.lower()):
                manual_fake_score += 0.35
                if 'vaccine_misinformation' not in matches:
                    matches.append('vaccine_misinformation')
                logger.info("Manual detection: Vaccine misinformation detected")
            
            if re.search(r'climate.*funding|funding.*climate|climate.*money|money.*climate', original_text.lower()):
                manual_fake_score += 0.35
                if 'climate_funding_conspiracy' not in matches:
                    matches.append('climate_funding_conspiracy')
                logger.info("Manual detection: Climate funding conspiracy detected")
            
            if re.search(r'climate.*(hoax|fake|not real|created|invented)', original_text.lower()):
                manual_fake_score += 0.35
                if 'climate_denial_claim' not in matches:
                    matches.append('climate_denial_claim')
                logger.info("Manual detection: Climate change denial detected")
            
            if re.search(r'(miracle|magical).*(cure|heals).*cancer', original_text.lower()) or \
               re.search(r'cure.*cancer.*(days|weeks|overnight)', original_text.lower()):
                manual_fake_score += 0.40
                if 'miracle_cure_timeframe' not in matches:
                    matches.append('miracle_cure_timeframe')
                logger.info("Manual detection: Cancer miracle cure claim detected")
            
            if re.search(r'herb.*cure.*cancer|cancer.*cure.*herb', original_text.lower()) or \
               re.search(r'natural.*cure.*cancer|cancer.*cure.*natural', original_text.lower()):
                manual_fake_score += 0.35
                if 'miracle_natural_cure' not in matches:
                    matches.append('miracle_natural_cure')
                logger.info("Manual detection: Natural cancer cure claim detected")
            
            if re.search(r'cure.*no side effects|no side effects.*cure', original_text.lower()):
                manual_fake_score += 0.35
                if 'impossible_medical_claim' not in matches:
                    matches.append('impossible_medical_claim')
                logger.info("Manual detection: Impossible medical claim detected")
            
            if re.search(r'days.*cure.*cancer|cancer.*cure.*days|week.*cure.*cancer|cancer.*cure.*week', original_text.lower()):
                manual_fake_score += 0.40
                if 'quick_cure_claim' not in matches:
                    matches.append('quick_cure_claim')
                logger.info("Manual detection: Quick cancer cure claim detected")
            
            sequence = self.tokenizer.texts_to_sequences([cleaned])
            padded = pad_sequences(sequence, maxlen=MAX_LEN, truncating='post')
            tensor = torch.tensor(padded, dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                model_prob = self.model(tensor).item()
            
            adjusted_prob = min(model_prob + manual_fake_score, 0.99)
            
            logger.info(f"Original model probability: {model_prob}, Manual score: {manual_fake_score}, Adjusted: {adjusted_prob}")
            
            if manual_fake_score >= 0.30:  # Strong evidence of fake news
                adjusted_prob = max(adjusted_prob, 0.80)  # Ensure at least 80% fake probability
            elif manual_fake_score >= 0.15:  # Moderate evidence of fake news
                adjusted_prob = max(adjusted_prob, 0.65)  # Ensure at least 65% fake probability
                
            return {
                'prediction': 'Fake' if adjusted_prob > 0.5 else 'Real',
                'confidence': max(adjusted_prob, 1-adjusted_prob),
                'is_fake': adjusted_prob > 0.5,
                'raw_probability': model_prob,
                'adjusted_probability': adjusted_prob,
                'detected_patterns': matches if matches else []
            }
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                'error': str(e),
                'prediction': 'Error',
                'confidence': 0.0
            }

news_classifier = NewsClassifier()