# CreditRiskAnalyzer-ML

A **simple machine learning tool** to predict the **Probability of Default (PD)** for borrowers and calculate the **Expected Loss (EL)** on loans using a Logistic Regression model.

---

## Features
- Train a **Logistic Regression** model on borrower data.
- Predict **Probability of Default (PD)** for any loan application.
- Compute **Expected Loss (EL)** using:

\[
EL = PD \times LGD \times EAD
\]

Where:
- **PD** = Probability of Default (model output)  
- **LGD** = Loss Given Default (fixed at 90%, assuming 10% recovery)  
- **EAD** = Exposure at Default (loan amount outstanding)  