from app.agents.scope_agent import ScopeAgent


def main():
    source = input("Enter uploaded document filename: ").strip()

    agent = ScopeAgent()

    print("\n🧠 Running Scope & Deliverable Extraction Agent...\n")

    results = agent.extract_scope(source)

    print("=" * 60)
    print("AI-GENERATED PROJECT SCOPE")
    print("=" * 60)

    print(results["analysis"])


if __name__ == "__main__":
    main()