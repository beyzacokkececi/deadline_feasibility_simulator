# Capacity & Deadline Feasibility Simulator

A decision-support tool that answers a simple but critical question:

Is our roadmap actually feasible given our current developer capacity — even under optimal task allocation?

This simulator helps Product, Engineering, and Leadership teams visualize delivery risk before it becomes inevitable.

---

## Overview

The simulator:

- Takes a list of tasks with estimates and deadlines
- Simulates execution across a fixed set of developers
- Computes start and finish dates based on capacity constraints
- Calculates lateness and feasibility KPIs
- Provides optional solver-based task reallocation
- Visualizes the result as an interactive timeline

It turns “this feels risky” into:

“This plan is provably infeasible under current capacity.”

---

## Core Use Cases

- Roadmap feasibility review  
- Capacity planning discussions  
- Scope vs deadline trade-off analysis  
- Headcount justification  
- Executive-level delivery risk visualization  

---

## Key Concepts

### Planning Start Date

All scheduling is calculated from a configurable Planning Start Date.

This allows scenario testing such as:

- What if development starts next week?
- What if kickoff is delayed by three days?

There is no need to manually set start dates per task.

---

### Developer Capacity Model

- Each developer works sequentially (no parallel execution).
- Effective hours per day are configurable.
- Weekends can optionally be excluded.
- Tasks are executed in a defined order per developer.

---

### Feasibility Metrics

The simulator calculates:

- Start Date  
- Finish Date  
- Days Late  
- On-time vs Late status  
- Success Rate (%)  
- Maximum Delay (days)  

When Success Rate reaches 100%, all tasks are feasible under current assumptions.

---

## Solver Modes

The tool supports multiple assignment strategies.

### Manual Mode (Default)

Uses the assignment and execution order defined in the uploaded dataset.

### Minimize Total Weighted Lateness

Greedy solver that:

- Assigns tasks to developers
- Minimizes weighted lateness
- Takes task priority into account

Default priority weights:

- Highest → 5.0  
- High → 3.0  
- Medium → 1.0  
- Low → 0.5  

### Minimize Maximum Delay (Min–Max)

Greedy bottleneck minimization strategy that:

- Minimizes the worst delay across all tasks
- Helps prevent catastrophic overruns

Solver results generate a new assignment in-memory and do not overwrite the original dataset.

---

## Visualization

The simulator provides:

- Developer-lane timeline (Gantt-style view)
- Color-coded task status
  - Green → On-time  
  - Red → Late  
- Optional markers
  - Planning start  
  - Earliest missed deadline  
- KPI dashboard

---

## Required CSV Format

### Required Columns

- `task_name` (string)
- `estimate_md` (float, person-days, must be > 0)
- `deadline` (date, YYYY-MM-DD)

### Optional Columns

- `priority` (Highest, High, Medium, Low)
- `assigned_dev` (developer name)
- `dev_order` (integer > 0)

Example:

```csv
task_name,estimate_md,deadline,priority,assigned_dev,dev_order
Task A,5,2026-02-15,Highest,Ahmet,1
Task B,3,2026-02-20,High,Mehmet,1
Task C,8,2026-03-01,Medium,Ayşe,1
