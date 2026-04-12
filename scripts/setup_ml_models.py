#!/usr/bin/env python
# ============================================================
# scripts/setup_ml_models.py
#
# Quick setup script for ML models
# Run this to train all models
# ============================================================

import subprocess
import sys
import os
from pathlib import Path

def main():
    print("="*60)
    print("🚀 AFWAH Tracker ML Model Setup")
    print("="*60)
    
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    print("\n📋 Step 1: Checking dependencies...")
    
    required_packages = [
        'scikit-learn',
        'numpy',
        'scipy',
        'opencv-python',
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Installing missing packages: {', '.join(missing_packages)}")
        cmd = [sys.executable, '-m', 'pip', 'install'] + missing_packages
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print("❌ Failed to install packages")
            return False
    
    print("\n📚 Step 2: Creating models directory...")
    models_dir = project_root / "backend" / "ml" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    print(f"  ✅ Models directory: {models_dir}")
    
    print("\n🔧 Step 3: Training rumor detection models...")
    print("  (This may take 10-30 seconds...)\n")
    
    train_script = project_root / "scripts" / "train_rumor_detector.py"
    result = subprocess.run([sys.executable, str(train_script)])
    
    if result.returncode == 0:
        print("\n✅ Model training completed successfully!")
        
        # Verify models exist
        print("\n📦 Verifying trained models...")
        models_to_check = [
            'rf_rumor_model.pkl',
            'gb_rumor_model.pkl',
            'tfidf_vectorizer.pkl'
        ]
        
        for model_file in models_to_check:
            model_path = models_dir / model_file
            if model_path.exists():
                size_mb = model_path.stat().st_size / (1024 * 1024)
                print(f"  ✅ {model_file} ({size_mb:.2f} MB)")
            else:
                print(f"  ❌ {model_file} NOT FOUND")
                return False
        
        print("\n" + "="*60)
        print("🎉 ML MODELS READY FOR USE!")
        print("="*60)
        print("\nNext steps:")
        print("1. Restart the backend server")
        print("2. Models will be automatically loaded")
        print("3. Test with: python scripts/test_ml_models.py")
        print("\nDocumentation:")
        print("  See: ML_IMPROVEMENTS_README.md")
        print("="*60)
        
        return True
    else:
        print("\n❌ Model training failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
