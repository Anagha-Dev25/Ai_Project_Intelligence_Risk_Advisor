import csv
import io
import re
from typing import Dict, Any, List, Optional
import datetime


def export_risk_register_csv(documentation_results: Optional[Dict[str, Any]]) -> str:
    """
    Parse risk register text from DocumentationAgent results into a clean CSV string.
    """
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Risk ID", "Risk Title", "Reason", "Impact", "Probability", "Severity", "Mitigation", "Evidence"])

    if not documentation_results:
        return output.getvalue()

    text = documentation_results.get("risk_register", "")
    if not text or "no evidence-based risk" in text.lower():
        return output.getvalue()

    # Pattern to match numbered risks
    risk_blocks = re.split(r"\n(?=\d+\.\s*Risk:)", text)
    risk_id = 1

    for block in risk_blocks:
        if "Risk:" not in block:
            continue

        title_match = re.search(r"Risk:\s*(.*?)(?=\n\s*(?:Reason|Impact|Probability|Severity|Mitigation|Evidence):|\Z)", block, re.DOTALL)
        reason_match = re.search(r"Reason:\s*(.*?)(?=\n\s*(?:Impact|Probability|Severity|Mitigation|Evidence):|\Z)", block, re.DOTALL)
        impact_match = re.search(r"Impact:\s*(.*?)(?=\n\s*(?:Probability|Severity|Mitigation|Evidence):|\Z)", block, re.DOTALL)
        prob_match = re.search(r"Probability:\s*([^\n]+)", block)
        sev_match = re.search(r"Severity:\s*([^\n]+)", block)
        mit_match = re.search(r"Mitigation:\s*(.*?)(?=\n\s*(?:Evidence):|\Z)", block, re.DOTALL)
        ev_match = re.search(r"Evidence:\s*(.*?)$", block, re.DOTALL)

        title = title_match.group(1).strip() if title_match else "Unspecified Risk"
        reason = reason_match.group(1).strip() if reason_match else ""
        impact = impact_match.group(1).strip() if impact_match else ""
        prob = prob_match.group(1).strip() if prob_match else "Medium"
        sev = sev_match.group(1).strip() if sev_match else "Medium"
        mit = mit_match.group(1).strip() if mit_match else ""
        ev = ev_match.group(1).strip() if ev_match else ""

        writer.writerow([f"RSK-{risk_id:03d}", title, reason, impact, prob, sev, mit, ev])
        risk_id += 1

    return output.getvalue()


def export_action_items_csv(documentation_results: Optional[Dict[str, Any]]) -> str:
    """
    Parse action items text from DocumentationAgent results into a clean CSV string.
    """
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Action ID", "Action Item", "Owner", "Deadline", "Priority", "Evidence"])

    if not documentation_results:
        return output.getvalue()

    text = documentation_results.get("action_items", "")
    if not text or "no evidence-based action" in text.lower():
        return output.getvalue()

    action_blocks = re.split(r"\n(?=\d+\.\s*Action:)", text)
    act_id = 1

    for block in action_blocks:
        if "Action:" not in block:
            continue

        desc_match = re.search(r"Action:\s*(.*?)(?=\n\s*(?:Owner|Deadline|Priority|Evidence):|\Z)", block, re.DOTALL)
        owner_match = re.search(r"Owner:\s*([^\n]+)", block)
        deadline_match = re.search(r"Deadline:\s*([^\n]+)", block)
        prio_match = re.search(r"Priority:\s*([^\n]+)", block)
        ev_match = re.search(r"Evidence:\s*(.*?)$", block, re.DOTALL)

        desc = desc_match.group(1).strip() if desc_match else "Unspecified Action"
        owner = owner_match.group(1).strip() if owner_match else "Unassigned"
        deadline = deadline_match.group(1).strip() if deadline_match else "Not specified"
        prio = prio_match.group(1).strip() if prio_match else "Medium"
        ev = ev_match.group(1).strip() if ev_match else ""

        writer.writerow([f"ACT-{act_id:03d}", desc, owner, deadline, prio, ev])
        act_id += 1

    return output.getvalue()


def export_user_stories_csv(documentation_results: Optional[Dict[str, Any]]) -> str:
    """
    Parse user stories from DocumentationAgent results into a clean CSV string.
    """
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Story ID", "User Story", "Priority", "Evidence"])

    if not documentation_results:
        return output.getvalue()

    text = documentation_results.get("user_stories", "")
    if not text or "no evidence-based user story" in text.lower():
        return output.getvalue()

    story_blocks = re.split(r"\n(?=\d+\.\s*User Story:)", text)
    story_id = 1

    for block in story_blocks:
        if "User Story:" not in block:
            continue

        story_match = re.search(r"User Story:\s*(.*?)(?=\n\s*(?:Priority|Evidence):|\Z)", block, re.DOTALL)
        prio_match = re.search(r"Priority:\s*([^\n]+)", block)
        ev_match = re.search(r"Evidence:\s*(.*?)$", block, re.DOTALL)

        story = story_match.group(1).strip() if story_match else "Unspecified Story"
        prio = prio_match.group(1).strip() if prio_match else "Medium"
        ev = ev_match.group(1).strip() if ev_match else ""

        writer.writerow([f"US-{story_id:03d}", story, prio, ev])
        story_id += 1

    return output.getvalue()


def generate_executive_report_markdown(
    project_sources: List[str],
    health_results: Optional[Dict[str, Any]],
    analysis_results: Optional[Dict[str, Any]],
    documentation_results: Optional[Dict[str, Any]]
) -> str:
    """
    Generate a comprehensive Executive Project Intelligence & Risk Report in Markdown format.
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    sources_str = ", ".join(project_sources) if project_sources else "None"

    report = f"""# 📊 ProjectIQ Executive Project Intelligence Report
**Generated:** {now}  
**Active Project Documents:** `{sources_str}`  

---

## 1. Executive Summary & Health Dashboard
"""

    if health_results:
        report += f"""
- **Overall Project Health Score:** **{health_results.get('overall_score', 'N/A')}/100**
- **Health Status:** **{health_results.get('status', 'N/A')}**

### Dimension Breakdown
| Dimension | Score | Evaluation |
| :--- | :--- | :--- |
| **Scope Clarity** | {health_results.get('dimensions', {}).get('scope_clarity', 'N/A')}/100 | Documented deliverables, objectives & boundaries |
| **Timeline Health** | {health_results.get('dimensions', {}).get('timeline_health', 'N/A')}/100 | Schedule adherence, milestones & target dates |
| **Risk Status** | {health_results.get('dimensions', {}).get('risk_status', 'N/A')}/100 | Identified dependencies & delivery threats |
| **Blocker Status** | {health_results.get('dimensions', {}).get('blocker_status', 'N/A')}/100 | Active impediments preventing progress |

### Key Contributing Health Factors
"""
        for factor in health_results.get("health_factors", []):
            report += f"- {factor}\n"
    else:
        report += "\n*Project health calculation has not been executed.*\n"

    report += "\n---\n\n## 2. Project Scope & Architecture\n"
    if analysis_results and "scope" in analysis_results:
        report += analysis_results["scope"].get("analysis", "No scope analysis available.") + "\n"
    else:
        report += "*Scope extraction has not been executed.*\n"

    report += "\n---\n\n## 3. Risks & Delivery Forecast\n"
    if analysis_results and "risk" in analysis_results:
        report += analysis_results["risk"].get("analysis", "No risk analysis available.") + "\n"
    else:
        report += "*Risk analysis has not been executed.*\n"

    report += "\n---\n\n## 4. Blockers, Decisions & Responsibilities\n"
    if analysis_results and "blocker" in analysis_results:
        report += analysis_results["blocker"].get("analysis", "No blocker analysis available.") + "\n"
    else:
        report += "*Blocker analysis has not been executed.*\n"

    report += "\n---\n\n## 5. Agile Documentation & Action Items\n"
    if documentation_results:
        report += f"""### Generated User Stories
{documentation_results.get('user_stories', 'None')}

### Risk Register
{documentation_results.get('risk_register', 'None')}

### Traceable Action Items
{documentation_results.get('action_items', 'None')}
"""
    else:
        report += "*Documentation generation has not been executed.*\n"

    report += "\n---\n*Report generated automatically by ProjectIQ • AI-Driven Enterprise Intelligence Platform*\n"
    return report
