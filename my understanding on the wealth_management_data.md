My Understanding of the Data

When I went through these tables, I realized they can be split into three simple layers:

Sales Pipeline → What happens before a deal is closed.

Core Operations → Once a deal becomes official, this is the main system of record.

Business Intelligence → Summaries and reports built on top of operations to help us analyze performance.

Layer 1: Sales Pipeline

1. wealth_management_opportunities
This is basically our sales funnel. It tracks every deal from start to finish (won or lost).

Helps me see: how the pipeline looks this quarter, what stage deals are stuck at, and why we’re losing deals.

Once an opportunity is won, it links to the core system through a contract number.

2. wealth_management_activities
This is like the logbook of everything sales did — every call, email, or meeting.

Useful for checking: how many calls were made, outcomes of those calls, or the full history of a specific opportunity.

Always tied back to the opportunity ID.

Layer 2: Core Operations

3. wealth_management_policy
This is the master file of all contracts/policies. It’s the center of everything.

Tells me: how many active policies we have, their face value, and which ones are expiring soon.

4. wealth_management_agents
This is the directory of all agents.

Answers things like: who is the agent for a given policy, which branch/division they belong to, and their status/rank.

5. wealth_management_clients
This shows the relationship of each client to a policy.

Example: who is the policy owner, who are the beneficiaries, or who has Power of Attorney.

6. wealth_management_transactions
This is the money side. Every payment, commission, or transaction linked to a policy is here.

Helps check premium payments, commission payouts, and full payment history.

Layer 3: Business Intelligence

7. wealth_management_pcpg_retention
This feels like a client summary dashboard.

Useful for spotting: top clients by AUM, retention risk, client tenure, etc.

8. wealth_management_rpt_agents
This is the agent performance report (monthly).

Shows things like: YTD production credits, new hire performance, and active agent counts.

9. remote_advise_sales_reporting
This is a special report just for remote advisors.

Helps track: how much revenue remote agents bring, which products they’re selling most, and their effectiveness.

10. wealth_management_client_metrics
This is the most detailed client profile — it even mixes in external data.

Answers: how much of a client’s total wealth we manage (share of wallet), what external assets they hold, and how we can segment them better.
