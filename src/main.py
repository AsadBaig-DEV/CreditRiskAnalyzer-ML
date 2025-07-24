from calculator import expected_loss
from model import train_model

def main():
    print("===Credit Risk Analyzer===")
    print("1. Calculate Expected Loss")
    print("2. ReTrain Model")

    choice = input("Select an option")

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
    
    else:
        print("Invalid choice. Please select 1 or 2.")

if __name__ == "__main__":
    main()
