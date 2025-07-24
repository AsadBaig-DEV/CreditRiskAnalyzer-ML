from model import load_model

def expected_loss(loan_features, recovery_rate=0.1):
    """
    loan_features: [credit_lines_outstanding, loan_amt_outstanding, total_debt_outstanding, 
                    income, years_employed, fico_score]
    """

    model, scaler = load_model()
    loan_scaled = scaler.transform([loan_features])
    pd = model.predict_proba(loan_scaled)[0][1]
    lgd = 1 - recovery_rate
    ead = loan_features[1]
    el = pd * lgd * ead
    return {
        "PD": pd,
        "LGD": lgd,
        "EAD": ead,
        "Expected Loss": el
    }