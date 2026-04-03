## Contributing to PFAD

Thank you for your interest in contributing to the Personal Finance Anomaly Detector! To ensure the project maintains its high standards of calm intelligence, rigorous ML performance, and premium UI integration, please follow these guidelines:

### 1. Architecture Patterns
- **Core Logic:** All data transformation, machine learning models, and rule extractions must remain in the `/src` folder. `app.py` is strictly for presentation, layout, and orchestration.
- **Ensemble ML Rules:** If you add anomaly detection signals (like our Isolation Forest and Z-Score models), ensure they expose `.fit_predict` and that confidence voting remains integrated.

### 2. Testing Constraints
- We enforce strict adversarial testing across all modules.
- Write tests for all new features under the `tests/` directory.
- Run `pytest` and verify coverage before submitting any PRs. Any degradation in coverage is not accepted.

### 3. Style & Tone Guidelines
- Maintain the "Calm Intelligence" design philosophy. Avoid loud colors (reds/yellows) unless an explicitly critical anomaly is detected.
- Never use forceful warnings like "ACTION REQUIRED." Instead, provide structural "Suggested Adjustments."

### 4. Submitting a Pull Request
- Create a feature branch (`git checkout -b feature/your-feature`).
- Ensure `config.yaml` example files are maintained without leaking actual credentials.
- Outline your design reasoning extensively in the PR context.
