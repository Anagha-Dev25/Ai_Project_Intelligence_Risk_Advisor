from app.agents.risk_agent import RiskAgent


def main():
    source = input("Enter uploaded document filename: ").strip()

    agent = RiskAgent()

    print("\n⚠️ Running Risk Detection & Delivery Forecasting Agent...\n")

    results = agent.analyze_risks(source)

    print("=" * 60)
    print("AI-GENERATED RISK & DELIVERY ANALYSIS")
    print("=" * 60)

    print(results["analysis"])


if __name__ == "__main__":
    main()