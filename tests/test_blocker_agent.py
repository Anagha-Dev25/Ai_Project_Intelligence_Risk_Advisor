import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.agents.blocker_agent import BlockerAgent


def main():
    source = input("Enter uploaded document filename: ").strip()

    agent = BlockerAgent()

    print("\n🚧 Running Blocker & Action Item Identification Agent...\n")

    results = agent.identify_blockers(source)

    print("=" * 60)
    print("AI-GENERATED BLOCKER & ACTION ITEM ANALYSIS")
    print("=" * 60)

    print(results["analysis"])


if __name__ == "__main__":
    main()