# ============================================================
# scripts/train_rumor_detector.py
#
# Training script for improved rumor detection model
# Trains on labeled dataset to improve classification
# ============================================================

import json
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pickle

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
except ImportError:
    print("⚠️ Install scikit-learn: pip install scikit-learn")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RumorDetectorTrainer:
    """Train improved rumor detection models"""
    
    def __init__(self, data_path: str = "data/rumor_training_data.json"):
        self.data_path = Path(data_path)
        self.model_path = Path("backend/ml/models")
        self.model_path.mkdir(exist_ok=True)
        
        self.training_data = []
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        self.rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
        self.gb_model = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
    
    def load_training_data(self) -> bool:
        """Load training data from JSON file"""
        try:
            logger.info(f"Loading training data from {self.data_path}...")
            
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.training_data = data.get('training_data', [])
            
            if not self.training_data:
                logger.error("No training data found")
                return False
            
            logger.info(f"✅ Loaded {len(self.training_data)} training samples")
            
            # Print category distribution
            categories = {}
            for sample in self.training_data:
                cat = sample.get('category', 'unknown')
                categories[cat] = categories.get(cat, 0) + 1
            
            logger.info("Category distribution:")
            for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
                logger.info(f"  {cat}: {count} samples")
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to load training data: {e}")
            return False
    
    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare and split training data"""
        
        logger.info("Preparing training data...")
        
        texts = [sample['text'] for sample in self.training_data]
        labels = np.array([sample['label'] for sample in self.training_data])
        
        # Convert continuous labels (0, 0.5, 1.0) to discrete classes (0, 1)
        # Treat 0.5 (mixed/uncertain) as 1 (needs review/accurate side)
        discrete_labels = np.round(labels).astype(int)
        
        # Vectorize texts
        X = self.vectorizer.fit_transform(texts)
        
        logger.info(f"🔢 Feature space: {X.shape}")
        logger.info(f"📊 Label distribution (original): {np.unique(labels, return_counts=True)}")
        logger.info(f"📊 Label distribution (discrete): {np.unique(discrete_labels, return_counts=True)}")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, discrete_labels,
            test_size=0.2,
            random_state=42,
            stratify=discrete_labels
        )
        
        logger.info(f"✅ Split: {X_train.shape[0]} train, {X_test.shape[0]} test")
        
        return X_train, X_test, y_train, y_test
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """Train multiple models and evaluate"""
        
        logger.info("\n" + "="*60)
        logger.info("TRAINING RANDOM FOREST MODEL")
        logger.info("="*60)
        
        # Train Random Forest
        self.rf_model.fit(X_train, y_train)
        rf_score = self.rf_model.score(X_test, y_test)
        rf_train_score = self.rf_model.score(X_train, y_train)
        
        logger.info(f"📊 Random Forest Train Accuracy: {rf_train_score:.4f}")
        logger.info(f"📊 Random Forest Test Accuracy: {rf_score:.4f}")
        
        # Cross-validation
        rf_cv_scores = cross_val_score(self.rf_model, X_train, y_train, cv=5)
        logger.info(f"📊 Random Forest CV Scores: {rf_cv_scores}")
        logger.info(f"📊 Random Forest CV Mean: {rf_cv_scores.mean():.4f} (+/- {rf_cv_scores.std():.4f})")
        
        logger.info("\n" + "="*60)
        logger.info("TRAINING GRADIENT BOOSTING MODEL")
        logger.info("="*60)
        
        # Train Gradient Boosting
        self.gb_model.fit(X_train, y_train)
        gb_score = self.gb_model.score(X_test, y_test)
        gb_train_score = self.gb_model.score(X_train, y_train)
        
        logger.info(f"📊 Gradient Boosting Train Accuracy: {gb_train_score:.4f}")
        logger.info(f"📊 Gradient Boosting Test Accuracy: {gb_score:.4f}")
        
        # Cross-validation
        gb_cv_scores = cross_val_score(self.gb_model, X_train, y_train, cv=5)
        logger.info(f"📊 Gradient Boosting CV Scores: {gb_cv_scores}")
        logger.info(f"📊 Gradient Boosting CV Mean: {gb_cv_scores.mean():.4f} (+/- {gb_cv_scores.std():.4f})")
        
        logger.info("\n" + "="*60)
        logger.info("DETAILED EVALUATION")
        logger.info("="*60)
        
        # Get predictions
        rf_pred = self.rf_model.predict(X_test)
        gb_pred = self.gb_model.predict(X_test)
        
        logger.info("\n🌳 RANDOM FOREST Classification Report:")
        logger.info(classification_report(y_test, rf_pred, digits=4))
        
        logger.info("\n📈 GRADIENT BOOSTING Classification Report:")
        logger.info(classification_report(y_test, gb_pred, digits=4))
        
        # Confusion matrices
        logger.info("\n🌳 Random Forest Confusion Matrix:")
        logger.info(confusion_matrix(y_test, rf_pred))
        
        logger.info("\n📈 Gradient Boosting Confusion Matrix:")
        logger.info(confusion_matrix(y_test, gb_pred))
        
        return rf_score, gb_score
    
    def save_models(self):
        """Save trained models"""
        
        try:
            logger.info("\n💾 Saving models...")
            
            # Save Random Forest
            with open(self.model_path / 'rf_rumor_model.pkl', 'wb') as f:
                pickle.dump(self.rf_model, f)
            
            # Save Gradient Boosting
            with open(self.model_path / 'gb_rumor_model.pkl', 'wb') as f:
                pickle.dump(self.gb_model, f)
            
            # Save Vectorizer
            with open(self.model_path / 'tfidf_vectorizer.pkl', 'wb') as f:
                pickle.dump(self.vectorizer, f)
            
            logger.info("✅ Models saved successfully!")
            logger.info(f"   Random Forest: {self.model_path}/rf_rumor_model.pkl")
            logger.info(f"   Gradient Boosting: {self.model_path}/gb_rumor_model.pkl")
            logger.info(f"   Vectorizer: {self.model_path}/tfidf_vectorizer.pkl")
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
            return False
    
    def evaluate_feature_importance(self):
        """Analyze feature importance"""
        
        logger.info("\n" + "="*60)
        logger.info("FEATURE IMPORTANCE ANALYSIS")
        logger.info("="*60)
        
        # Get feature names
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Random Forest feature importance
        rf_importance = self.rf_model.feature_importances_
        rf_top_indices = np.argsort(rf_importance)[-20:][::-1]
        
        logger.info("\n🌳 Top 20 Random Forest Features:")
        for idx in rf_top_indices:
            logger.info(f"  {feature_names[idx]}: {rf_importance[idx]:.4f}")
        
        # Gradient Boosting feature importance
        gb_importance = self.gb_model.feature_importances_
        gb_top_indices = np.argsort(gb_importance)[-20:][::-1]
        
        logger.info("\n📈 Top 20 Gradient Boosting Features:")
        for idx in gb_top_indices:
            logger.info(f"  {feature_names[idx]}: {gb_importance[idx]:.4f}")
    
    def print_summary(self, rf_score: float, gb_score: float):
        """Print training summary"""
        
        logger.info("\n" + "="*60)
        logger.info("TRAINING COMPLETE - SUMMARY")
        logger.info("="*60)
        logger.info(f"Training samples: {len(self.training_data)}")
        logger.info(f"Feature dimensions: {len(self.vectorizer.get_feature_names_out())}")
        logger.info(f"\nRandom Forest Test Accuracy: {rf_score:.4f} ({rf_score*100:.2f}%)")
        logger.info(f"Gradient Boosting Test Accuracy: {gb_score:.4f} ({gb_score*100:.2f}%)")
        logger.info(f"\nBest Model: {'Random Forest' if rf_score >= gb_score else 'Gradient Boosting'}")
        logger.info("="*60 + "\n")


def main():
    """Main training pipeline"""
    
    logger.info("🚀 Starting Rumor Detection Model Training...")
    logger.info("="*60)
    
    trainer = RumorDetectorTrainer()
    
    # Load data
    if not trainer.load_training_data():
        logger.error("❌ Failed to load training data")
        return False
    
    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data()
    
    # Train models
    rf_score, gb_score = trainer.train_models(X_train, X_test, y_train, y_test)
    
    # Analyze features
    trainer.evaluate_feature_importance()
    
    # Save models
    if not trainer.save_models():
        logger.error("❌ Failed to save models")
        return False
    
    # Print summary
    trainer.print_summary(rf_score, gb_score)
    
    logger.info("✅ Training pipeline completed successfully!")
    logger.info("\nNext steps:")
    logger.info("1. Models are saved and ready for use")
    logger.info("2. Use the trained models in backend/ml/improved_rumor_analyzer.py")
    logger.info("3. Add more training data to improve accuracy")
    logger.info("4. Retrain periodically with new data")
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
