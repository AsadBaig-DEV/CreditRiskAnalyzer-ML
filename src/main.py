from calculator import expected_loss
from model import train_model
from quantizer import generate_buckets, assign_rating
import pandas as pd

def main():
    print("=== Credit Risk Analyzer ===")
    print("1. Calculate Expected Loss for a Loan")
    print("2. Retrain Model")
    print("3. Generate FICO Rating Buckets")
    choice = input("Select an option: ")

    if choice == "1":
        credit_lines_outstanding = float(input("Credit lines outstanding: "))
        loan_amt_outstanding = float(input("Loan amount outstanding: "))
        total_debt_outstanding = float(input("Total debt outstanding: "))
        income = float(input("Annual income: "))
        years_employed = float(input("Years employed: "))
        fico_score = float(input("FICO score: "))

        features = [credit_lines_outstanding, loan_amt_outstanding, total_debt_outstanding,
                    income, years_employed, fico_score]
        result = expected_loss(features)
        print("\n=== Expected Loss Calculation ===")
        print(f"Probability of Default (PD): {result['PD']:.3f}")
        print(f"Loss Given Default (LGD): {result['LGD']:.2f}")
        print(f"Exposure at Default (EAD): {result['EAD']:.2f}")
        print(f"Expected Loss: {result['Expected Loss']:.2f}")

    elif choice == "2":
        train_model()
        print("Model retrained successfully.")

    elif choice == "3":
        df = pd.read_csv("data/loan_data.csv")
        n_buckets = int(input("Enter number of buckets: "))
        method = input("Method (mse/loglikelihood): ").lower()
        buckets = generate_buckets(df, n_buckets=n_buckets, method=method)
        print("\n=== FICO Rating Buckets ===")
        for idx, (low, high) in enumerate(buckets, start=1):
            print(f"Rating {idx}: {int(low)} - {int(high)}")
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()
