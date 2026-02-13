import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from io import StringIO
import numpy as np
from copy import deepcopy

# ========================================================================
# SOLVER ALGORITHMS
# ========================================================================

def solver_minimize_total_days_late(df, developer_names, effective_hours, include_weekends=False,
                                    priority_col="priority", priority_weights=None, planning_start_date=None):
    if df.empty:
        return df.copy()

    result_df = df.copy()

    # Default weights
    if priority_weights is None:
        priority_weights = {
            "highest": 5.0,
            "high": 3.0,
            "medium": 1.0,
            "low": 0.5,
        }

    def priority_weight(p):
        if p is None or (isinstance(p, float) and pd.isna(p)):
            return 1.0
        p = str(p).strip().lower()
        return float(priority_weights.get(p, 1.0))

    # Sort by deadline, then priority (higher first), then name for determinism
    # (priority sorting doesn't change the objective, but helps greedy behave better)
    sorted_tasks = (
        result_df.assign(_pw=result_df[priority_col].apply(priority_weight) if priority_col in result_df.columns else 1.0)
                 .sort_values(['deadline', '_pw', 'task_name'], ascending=[True, False, True])
                 .reset_index(drop=True)
    )

    if planning_start_date is None:
        planning_start_date = datetime.now().date()

    start_date = datetime.combine(planning_start_date, datetime.min.time())

    dev_available = {name: start_date for name in developer_names}

    def calculate_finish_time(start_time, estimate_md, effective_hours, include_weekends):
        duration_days = estimate_md / (effective_hours / 8.0)

        if not include_weekends and start_time.weekday() >= 5:
            start_time += timedelta(days=7 - start_time.weekday())

        if include_weekends:
            return start_time + timedelta(days=duration_days)

        current_date = start_time
        days_added = 0.0

        # Supports fractional durations by adding 1 workday at a time; last step may overshoot slightly.
        # If you want exact fractional weekday handling, we can refine this later.
        while days_added < duration_days:
            current_date += timedelta(days=1)
            if current_date.weekday() < 5:
                days_added += 1.0

        return current_date

    assignments = []
    orders = {name: 1 for name in developer_names}

    for _, task in sorted_tasks.iterrows():
        best_dev = None
        best_score = float('inf')
        best_lateness = None

        w = priority_weight(task[priority_col]) if priority_col in sorted_tasks.columns else 1.0

        for dev_name in developer_names:
            finish_time = calculate_finish_time(
                dev_available[dev_name],
                task['estimate_md'],
                effective_hours,
                include_weekends
            )

            deadline = pd.to_datetime(task['deadline'])
            lateness = max(0, (finish_time - deadline).days)

            score = w * lateness

            # determinism: score -> lateness -> dev_name
            if (
                score < best_score
                or (score == best_score and lateness < best_lateness)
                or (score == best_score and lateness == best_lateness and (best_dev is None or dev_name < best_dev))
            ):
                best_score = score
                best_lateness = lateness
                best_dev = dev_name

        assignments.append({
            'task_name': task['task_name'],
            'assigned_dev': best_dev,
            'dev_order': orders[best_dev]
        })

        finish_time = calculate_finish_time(
            dev_available[best_dev],
            task['estimate_md'],
            effective_hours,
            include_weekends
        )
        dev_available[best_dev] = finish_time
        orders[best_dev] += 1

    assignment_df = pd.DataFrame(assignments)
    result_df = result_df.drop(['assigned_dev', 'dev_order'], axis=1, errors='ignore')
    result_df = result_df.merge(assignment_df, on='task_name', how='left')
    return result_df



def solver_minimize_maximum_delay(df, developer_names, effective_hours, include_weekends=False, planning_start_date=None):
    """
    Greedy solver that minimizes the maximum delay (min-max lateness).
    
    Strategy: Sort tasks by slack time (deadline - estimated_duration), then assign each task
    to the developer who results in the smallest maximum delay across all tasks so far.
    This is also known as the "bottleneck" minimization approach.
    """
    if df.empty:
        return df.copy()
    
    # Create a copy to modify
    result_df = df.copy()
    
    # Calculate slack time for each task (deadline - duration from now)
    if planning_start_date is None:
        planning_start_date = datetime.now().date()

    start_date = datetime.combine(planning_start_date, datetime.min.time())

    result_df['_slack'] = result_df.apply(
        lambda row: (pd.to_datetime(row['deadline']) - start_date).days - (row['estimate_md'] / (effective_hours / 8.0)),
        axis=1
    )
    
    # Sort by slack (least slack first), then deadline, then task name for determinism
    sorted_tasks = result_df.sort_values(['_slack', 'deadline', 'task_name']).reset_index(drop=True)
    
    # Track when each developer is available
    dev_available = {name: start_date for name in developer_names}
    
    def calculate_finish_time(start_time, estimate_md, effective_hours, include_weekends):
        """Calculate when a task would finish if started at start_time"""
        duration_days = estimate_md / (effective_hours / 8.0)
        
        # If not including weekends and start is on weekend, move to Monday
        if not include_weekends and start_time.weekday() >= 5:
            days_to_monday = 7 - start_time.weekday()
            start_time += timedelta(days=days_to_monday)
        
        # Add working days
        if include_weekends:
            return start_time + timedelta(days=duration_days)
        
        current_date = start_time
        days_added = 0
        
        while days_added < duration_days:
            current_date += timedelta(days=1)
            if current_date.weekday() < 5:
                days_added += 1
        
        return current_date
    
    # Assign each task to minimize the maximum delay
    assignments = []
    orders = {name: 1 for name in developer_names}
    current_max_delay = 0
    
    for idx, task in sorted_tasks.iterrows():
        deadline = pd.to_datetime(task['deadline'])
        best_dev = None
        best_new_max_delay = float('inf')
        best_task_delay = float('inf')
        
        # Try each developer and see what the maximum delay would be
        for dev_name in developer_names:
            finish_time = calculate_finish_time(
                dev_available[dev_name],
                task['estimate_md'],
                effective_hours,
                include_weekends
            )
            
            # Calculate this task's delay
            task_delay = max(0, (finish_time - deadline).days)
            
            # What would be the new maximum delay if we assign to this developer?
            new_max_delay = max(current_max_delay, task_delay)
            
            # Choose developer that results in smallest maximum delay
            # Tie-breaking: prefer developer with smallest task delay, then by name
            if (new_max_delay < best_new_max_delay or 
                (new_max_delay == best_new_max_delay and 
                 (task_delay < best_task_delay or 
                  (task_delay == best_task_delay and dev_name < best_dev)))):
                best_new_max_delay = new_max_delay
                best_dev = dev_name
                best_task_delay = task_delay
        
        # Assign task to best developer
        assignments.append({
            'task_name': task['task_name'],
            'assigned_dev': best_dev,
            'dev_order': orders[best_dev]
        })
        
        # Update developer availability and current max delay
        finish_time = calculate_finish_time(
            dev_available[best_dev],
            task['estimate_md'],
            effective_hours,
            include_weekends
        )
        dev_available[best_dev] = finish_time
        orders[best_dev] += 1
        current_max_delay = best_new_max_delay
    
    # Create assignment dataframe and merge
    assignment_df = pd.DataFrame(assignments)
    result_df = result_df.drop(['assigned_dev', 'dev_order', '_slack'], axis=1, errors='ignore')
    result_df = result_df.merge(assignment_df, on='task_name', how='left')
    
    return result_df



# ========================================================================
# SIMULATION FUNCTION
# ========================================================================

def simulate_schedule(df, developer_names, effective_hours, include_weekends=False, planning_start_date=None):
    """
    Simulate task execution based on assignments and ordering.
    Returns dataframe with start dates, finish dates, and status.
    """
    # Validate inputs
    if df.empty:
        return pd.DataFrame(columns=[
            'task_name', 'estimate_md', 'assigned_dev', 'dev_order',
            'deadline', 'start_date', 'finish_date', 'is_late', 'days_late', 'status'
        ])
    
    # Validate estimates are positive
    invalid_estimates = df[df['estimate_md'] <= 0]
    if not invalid_estimates.empty:
        st.error(f"⚠️ Invalid estimates found: {', '.join(invalid_estimates['task_name'].tolist())}. All estimates must be greater than 0.")
        return pd.DataFrame()
    
    if planning_start_date is None:
        planning_start_date = datetime.now().date()

    start_date = datetime.combine(planning_start_date, datetime.min.time())

    results = []
    
    # Helper function to add working days
    def add_working_days(start_date, days_to_add, include_weekends):
        if include_weekends:
            return start_date + timedelta(days=days_to_add)
        
        current_date = start_date
        days_added = 0
        
        while days_added < days_to_add:
            current_date += timedelta(days=1)
            # Skip weekends (5=Saturday, 6=Sunday)
            if current_date.weekday() < 5:
                days_added += 1
        
        return current_date
    
    # Track when each developer is available
    dev_available = {name: start_date for name in developer_names}
    
    # FIXED: Deterministic tie-breaking using deadline and task_name
    df_sorted = df.copy()
    df_sorted = df_sorted.sort_values([
        'assigned_dev',
        'dev_order',
        'deadline',      # Secondary: earlier deadline executes first
        'task_name'      # Tertiary: alphabetical
    ])
    
    for _, task in df_sorted.iterrows():
        dev_name = task['assigned_dev']
        estimate_md = float(task['estimate_md'])
        
        # Skip if developer name not in list
        if dev_name not in dev_available:
            st.warning(f"⚠️ Task '{task['task_name']}' assigned to unknown developer '{dev_name}'. Skipping.")
            continue
        
        # Calculate duration in calendar days
        duration_days = estimate_md / (effective_hours / 8.0)
        
        # Task starts when developer is available
        task_start = dev_available[dev_name]
        
        # If not including weekends and start is on weekend, move to Monday
        if not include_weekends and task_start.weekday() >= 5:
            days_to_monday = 7 - task_start.weekday()
            task_start += timedelta(days=days_to_monday)
        
        # Calculate finish date accounting for working days
        task_finish = add_working_days(task_start, duration_days, include_weekends)
        
        # Update developer availability
        dev_available[dev_name] = task_finish
        
        # Determine if late
        deadline = pd.to_datetime(task['deadline'])
        is_late = task_finish > deadline
        days_late = (task_finish - deadline).days if is_late else 0
        
        results.append({
            'task_name': task['task_name'],
            'estimate_md': estimate_md,
            'assigned_dev': dev_name,
            'dev_order': task['dev_order'],
            'deadline': deadline,
            'start_date': task_start,
            'finish_date': task_finish,
            'is_late': is_late,
            'days_late': days_late,
            'status': 'Late 🔴' if is_late else 'On-time 🟢'
        })
    
    return pd.DataFrame(results)


# ========================================================================
# STREAMLIT APP
# ========================================================================

# Page configuration
st.set_page_config(
    page_title="Capacity vs Deadline Feasibility Simulator",
    page_icon="🫨",
    layout="wide"
)

# Fun custom CSS
st.markdown("""
<style>
    /* Fun header styling */
    .main h1 {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem !important;
        font-weight: 800 !important;
        animation: slideIn 0.5s ease-out;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Fun metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }
    
    /* Colorful buttons */
    .stButton>button {
        border-radius: 20px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    /* Fun dividers */
    hr {
        margin: 2rem 0;
        border: none;
        height: 3px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        border-radius: 3px;
    }
    
    /* Pulse animation for critical items */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🫨 Yetişebiliyor muyuz Simulator")
st.markdown("""
This tool helps determine whether a set of engineering tasks can realistically be completed 
by their deadlines given current team capacity.

**Core Assumptions (Best-Case Scenario):**
- Each developer works on only one task at a time
- Each task is executed by a single developer
- All tasks are available to start immediately
""")

st.markdown("<hr>", unsafe_allow_html=True)

# Initialize session state
if 'tasks_df' not in st.session_state:
    st.session_state.tasks_df = None
if 'manual_tasks_df' not in st.session_state:
    st.session_state.manual_tasks_df = None
if 'last_uploaded_file_id' not in st.session_state:
    st.session_state.last_uploaded_file_id = None
if 'num_developers' not in st.session_state:
    st.session_state.num_developers = 3
if 'effective_hours' not in st.session_state:
    st.session_state.effective_hours = 8.0
if 'developer_names' not in st.session_state:
    st.session_state.developer_names = ['Berkcan', 'Yunus', 'Ekim']
if 'sort_column' not in st.session_state:
    st.session_state.sort_column = 'None'
if 'sort_direction' not in st.session_state:
    st.session_state.sort_direction = 'Ascending ⬆️'
if 'scheduling_mode' not in st.session_state:
    st.session_state.scheduling_mode = 'Manual'
if 'planning_start_date' not in st.session_state:
    st.session_state.planning_start_date = datetime.today()

# Sidebar: Team Configuration
st.sidebar.header("⚙️ Team Configuration")

num_developers = st.sidebar.number_input(
    "Number of Developers",
    min_value=1,
    max_value=20,
    value=st.session_state.num_developers,
    step=1,
    help="Total number of developers available"
)

# Update developer names list if count changes
if num_developers != st.session_state.num_developers:
    if num_developers > st.session_state.num_developers:
        # Add more developers
        for i in range(st.session_state.num_developers, num_developers):
            st.session_state.developer_names.append(f'Developer {i+1}')
    else:
        # FIXED: Handle tasks assigned to removed developers
        old_devs = set(st.session_state.developer_names)
        st.session_state.developer_names = st.session_state.developer_names[:num_developers]
        new_devs = set(st.session_state.developer_names)
        removed_devs = old_devs - new_devs
        
        # Reassign orphaned tasks
        if st.session_state.manual_tasks_df is not None and len(removed_devs) > 0:
            df = st.session_state.manual_tasks_df
            orphaned_mask = df['assigned_dev'].isin(removed_devs)
            
            if orphaned_mask.any():
                num_orphaned = orphaned_mask.sum()
                df.loc[orphaned_mask, 'assigned_dev'] = st.session_state.developer_names[0]
                st.session_state.manual_tasks_df = df
                st.sidebar.warning(f"⚠️ {num_orphaned} task(s) reassigned from removed developers to {st.session_state.developer_names[0]}")
    
    st.session_state.num_developers = num_developers

# Developer names input
st.sidebar.subheader("👥 Developer Names")
developer_names = []
for i in range(num_developers):
    name = st.sidebar.text_input(
        f"Developer {i+1}",
        value=st.session_state.developer_names[i] if i < len(st.session_state.developer_names) else f'Developer {i+1}',
        key=f'dev_name_{i}'
    )
    if not name or name.strip() == '':
        name = f'Developer {i+1}'
        st.sidebar.warning(f"Developer {i+1} name cannot be empty. Using default.")
    developer_names.append(name.strip())

st.session_state.developer_names = developer_names


effective_hours = st.sidebar.slider(
    "Effective Working Hours per Day",
    min_value=1.0,
    max_value=8.0,
    value=st.session_state.effective_hours,
    step=0.5,
    help="Actual productive hours per developer per day (accounting for meetings, context switching, etc.)"
)
st.session_state.effective_hours = effective_hours

planning_start_date = st.sidebar.date_input(
    "Planning Start Date",
    value=datetime.today(),
    help="All scheduling will assume work starts from this date (or next working day if weekends excluded)."
)

count_weekends = st.sidebar.checkbox(
    "Include Weekends as Working Days",
    value=False,
    help="If unchecked, only weekdays (Mon-Fri) count as working days"
)

work_days_per_week = 7 if count_weekends else 5
st.sidebar.markdown(f"""
**Current Configuration:**
- {num_developers} developer(s)
- {effective_hours} hours/day effective capacity
- {work_days_per_week} working days/week
- Total: {num_developers * effective_hours} person-hours/day
""")

st.sidebar.divider()

# File upload section
st.sidebar.header("📁 Upload Task Data")

uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=['csv'])

# Sample data generator
if st.sidebar.button("Generate Sample Data"):
    priorities = ['Highest', 'High', 'Medium', 'Low']
    sample_data = {
        'task_name': [f'Task {i+1}' for i in range(15)],
        'estimate_md': np.random.uniform(2, 10, 15).round(1),
        'deadline': [datetime.now() + timedelta(days=np.random.randint(14, 60)) for _ in range(15)],
        'priority': np.random.choice(priorities, 15)
    }
    st.session_state.manual_tasks_df = pd.DataFrame(sample_data)
    st.session_state.scheduling_mode = 'Manual'
    st.sidebar.success("Sample data generated!")
    st.rerun()

# Load data only when a *new* file is uploaded (not on every rerun)
if uploaded_file is not None:
    # Identify file: file_id (Streamlit 1.28+) or (name, size) so we don't reload on every rerun
    current_file_id = getattr(uploaded_file, 'file_id', (uploaded_file.name, uploaded_file.size))
    if current_file_id != st.session_state.last_uploaded_file_id:
        try:
            tasks_df = pd.read_csv(uploaded_file)
            
            # Validate required columns
            required_cols = ['task_name', 'estimate_md', 'deadline']
            missing_cols = [col for col in required_cols if col not in tasks_df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
            else:
                # Convert deadline to datetime
                try:
                    tasks_df['deadline'] = pd.to_datetime(tasks_df['deadline'])
                except Exception as e:
                    st.error(f"Invalid deadline format: {str(e)}. Please use YYYY-MM-DD format.")
                    tasks_df = None
                
                # Validate estimates during CSV upload
                if tasks_df is not None:
                    invalid_estimates = tasks_df[tasks_df['estimate_md'] <= 0]
                    if not invalid_estimates.empty:
                        st.error(f"❌ Invalid estimates in CSV: {', '.join(invalid_estimates['task_name'].tolist())}. All estimates must be greater than 0.")
                        tasks_df = None
                
                if tasks_df is not None:
                    st.session_state.manual_tasks_df = tasks_df
                    st.session_state.scheduling_mode = 'Manual'  # Reset to manual on upload
                    st.session_state.last_uploaded_file_id = current_file_id
                    st.sidebar.success(f"Loaded {len(tasks_df)} tasks")
                    st.rerun()
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
else:
    st.session_state.last_uploaded_file_id = None

# ========================================================================
# MAIN CONTENT - APPLY SCHEDULING MODE
# ========================================================================

if st.session_state.manual_tasks_df is not None:
    # Get the manual plan (from CSV or edits)
    manual_df = st.session_state.manual_tasks_df.copy()
    
    # Add default columns if not present in manual plan
    if 'priority' not in manual_df.columns:
        manual_df['priority'] = 'Medium'
    else:
        # Convert old numeric priorities to text if needed
        if manual_df['priority'].dtype in ['int64', 'int32', 'float64']:
            priority_map = {1: 'Highest', 2: 'High', 3: 'Medium', 4: 'Low'}
            manual_df['priority'] = manual_df['priority'].map(lambda x: priority_map.get(int(x), 'Medium'))
    
    if 'assigned_dev' not in manual_df.columns:
        # Simple round-robin assignment with names
        manual_df['assigned_dev'] = [developer_names[i % num_developers] for i in range(len(manual_df))]
    else:
        # Convert old numeric assignments to names if needed
        if manual_df['assigned_dev'].dtype in ['int64', 'int32']:
            manual_df['assigned_dev'] = manual_df['assigned_dev'].apply(
                lambda x: developer_names[int(x)-1] if 0 < x <= num_developers else developer_names[0]
            )
        # Validate assigned developers exist
        invalid_devs = ~manual_df['assigned_dev'].isin(developer_names)
        if invalid_devs.any():
            st.warning(f"⚠️ Some tasks assigned to unknown developers. Reassigning to {developer_names[0]}.")
            manual_df.loc[invalid_devs, 'assigned_dev'] = developer_names[0]
    
    if 'dev_order' not in manual_df.columns:
        # Order by priority within each developer
        manual_df['dev_order'] = manual_df.groupby('assigned_dev').cumcount() + 1
    
    # Apply scheduling mode to get the active dataframe
    if st.session_state.scheduling_mode == 'Manual':
        df = manual_df
        is_solver_mode = False
    elif st.session_state.scheduling_mode == 'Solver: Minimize Total Days Late':
        df = solver_minimize_total_days_late(manual_df, developer_names, effective_hours, count_weekends, planning_start_date)
        is_solver_mode = True
    elif st.session_state.scheduling_mode == 'Solver: Minimize Maximum Delay':
        df = solver_minimize_maximum_delay(manual_df, developer_names, effective_hours, count_weekends, planning_start_date)
        is_solver_mode = True
    else:
        df = manual_df
        is_solver_mode = False

    st.header("🔧 Interactive Task Planning")
    # ================================================================
    # SORTING CONTROLS (Manual and Solver modes)
    # ================================================================
    
    col1, col2, col3, col4 = st.columns([2, 2, 1, 3])
    with col1:
        sort_column = st.selectbox(
            "Sort by",
            options=['None', 'Task Name', 'Estimate (MD)', 'Deadline', 'Priority', 'Developer', 'Order', 'Finish Date', 'Days Late'],
            index=['None', 'Task Name', 'Estimate (MD)', 'Deadline', 'Priority', 'Developer', 'Order', 'Finish Date', 'Days Late'].index(st.session_state.sort_column),
        )
        st.session_state.sort_column = sort_column
        
    with col2:
        sort_direction = st.radio(
            "Direction",
            options=['Ascending ⬆️', 'Descending ⬇️'],
            index=0 if st.session_state.sort_direction == 'Ascending ⬆️' else 1,
            horizontal=True,
        )
        st.session_state.sort_direction = sort_direction
    with col3:
        if st.button("🔄 Apply Sort"):
            if sort_column != 'None':
                column_map = {
                    'Task Name': 'task_name',
                    'Estimate (MD)': 'estimate_md',
                    'Deadline': 'deadline',
                    'Priority': 'priority',
                    'Developer': 'assigned_dev',
                    'Order': 'dev_order',
                    'Finish Date': 'finish_date',
                    'Days Late': 'days_late'
                }
                
                # For priority, define custom order
                if sort_column == 'Priority':
                    priority_order = {'Highest': 0, 'High': 1, 'Medium': 2, 'Low': 3}
                    df['_priority_sort'] = df['priority'].map(priority_order)
                    col_to_sort = '_priority_sort'
                else:
                    col_to_sort = column_map.get(sort_column, 'task_name')
                
                ascending = sort_direction == 'Ascending ⬆️'
                
                # If sorting by feasibility columns, need to run simulation first
                if sort_column in ['Finish Date', 'Days Late']:
                    # Run quick simulation for sorting
                    temp_results = simulate_schedule(df, developer_names, effective_hours, count_weekends)
                    if not temp_results.empty:
                        df['finish_date'] = temp_results['finish_date']
                        df['days_late'] = temp_results['days_late']
                
                df = df.sort_values(col_to_sort, ascending=ascending)
                
                # Clean up temporary columns
                if '_priority_sort' in df.columns:
                    df = df.drop('_priority_sort', axis=1)
                if 'finish_date' in df.columns and sort_column == 'Finish Date':
                    df = df.drop(['finish_date', 'days_late'], axis=1, errors='ignore')
                elif 'finish_date' in df.columns:
                    df = df.drop(['finish_date', 'days_late'], axis=1, errors='ignore')
                
                df = df.reset_index(drop=True)
                st.session_state.manual_tasks_df = df
                st.rerun()
    
    # ================================================================
    # ADD NEW TASK BUTTON (only in Manual mode)
    # ================================================================
    
    if not is_solver_mode:
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("➕ Add New Task", use_container_width=True):
                # Calculate smart defaults
                existing_count = len(df)
                
                # Default order: next available for first developer
                dev_tasks = df[df['assigned_dev'] == developer_names[0]]
                next_order = dev_tasks['dev_order'].max() + 1 if len(dev_tasks) > 0 else 1
                
                # Default deadline: 30 days from latest existing deadline or today
                if len(df) > 0 and 'deadline' in df.columns:
                    latest_deadline = df['deadline'].max()
                    default_deadline = latest_deadline + timedelta(days=7)  # 1 week after latest
                else:
                    default_deadline = datetime.now() + timedelta(days=30)
                
                new_row = pd.DataFrame({
                    'task_name': [f'New Task {existing_count + 1}'],
                    'estimate_md': [3.0],
                    'deadline': [default_deadline],
                    'priority': ['Medium'],
                    'assigned_dev': [developer_names[0]],
                    'dev_order': [next_order]
                })
                df = pd.concat([df, new_row], ignore_index=True)
                st.session_state.manual_tasks_df = df
                st.rerun()
    
    # ================================================================
    # RUN SIMULATION AND DISPLAY TABLE
    # ================================================================
    
    # Run simulation to get feasibility data
    sim_results = simulate_schedule(df, developer_names, effective_hours, count_weekends, planning_start_date)
    
    # Only proceed if simulation succeeded
    if not sim_results.empty:
        # FIXED: Merge by task_name to handle row deletions/additions correctly
        merged_df = df.merge(
            sim_results[['task_name', 'start_date', 'finish_date', 'status', 'days_late']],
            on='task_name',
            how='left'
        )

        # Create editable or read-only dataframe based on mode
        if is_solver_mode:
            # Solver mode: read-only table
            st.dataframe(
                merged_df[['task_name', 'estimate_md', 'deadline', 'priority', 'assigned_dev', 'dev_order', 'start_date', 'finish_date', 'status', 'days_late']],
                column_config={
                    "task_name": st.column_config.TextColumn("Task Name", width="medium"),
                    "estimate_md": st.column_config.NumberColumn("Estimate (MD)", format="%.1f"),
                    "deadline": st.column_config.DateColumn("Deadline"),
                    "priority": st.column_config.TextColumn("Priority", width="small"),
                    "assigned_dev": st.column_config.TextColumn("Assigned Developer", width="medium"),
                    "dev_order": st.column_config.NumberColumn("Order", help="Execution order within developer's workload"),
                    "start_date": st.column_config.DateColumn("Start Date", help="Calculated start date"),
                    "finish_date": st.column_config.DateColumn("Finish Date", help="Calculated completion date"),
                    "status": st.column_config.TextColumn("Status", width="small", help="On-time or Late"),
                    "days_late": st.column_config.NumberColumn("Days Late", help="0 if on-time", format="%d")
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            # Manual mode: editable table
            edited_df = st.data_editor(
                merged_df[['task_name', 'estimate_md', 'deadline', 'priority', 'assigned_dev', 'dev_order', 'start_date', 'finish_date', 'status', 'days_late']],
                column_config={
                    "task_name": st.column_config.TextColumn("Task Name", width="medium"),
                    "estimate_md": st.column_config.NumberColumn(
                        "Estimate (MD)", 
                        min_value=0.1,
                        format="%.1f",
                        help="Must be greater than 0"
                    ),
                    "deadline": st.column_config.DateColumn("Deadline", min_value=datetime(2026, 1, 1), max_value=datetime(2026, 12, 31)),
                    "priority": st.column_config.SelectboxColumn(
                        "Priority",
                        options=['Highest', 'High', 'Medium', 'Low'],
                        width="small"
                    ),
                    "assigned_dev": st.column_config.SelectboxColumn(
                        "Assigned Developer",
                        options=developer_names,
                        width="medium"
                    ),
                    "dev_order": st.column_config.NumberColumn(
                        "Order", 
                        min_value=1, 
                        help="Execution order within developer's workload"
                    ),
                    "start_date": st.column_config.DateColumn("Start Date", disabled=True, help="Calculated start date"),
                    "finish_date": st.column_config.DateColumn("Finish Date", disabled=True, help="Calculated completion date"),
                    "status": st.column_config.TextColumn("Status", disabled=True, width="small", help="On-time or Late"),
                    "days_late": st.column_config.NumberColumn("Days Late", disabled=True, help="0 if on-time", format="%d")
                },
                hide_index=True,
                use_container_width=True,
                num_rows="dynamic",  # Allow deleting rows
                disabled=['start_date', 'finish_date', 'status', 'days_late'],
                column_order=['task_name', 'estimate_md', 'deadline', 'priority', 'assigned_dev', 'dev_order', 'start_date', 'finish_date', 'status', 'days_late']
            )
            
            # Update session state with only the editable columns (user edits)
            editable_data = edited_df[['task_name', 'estimate_md', 'deadline', 'priority', 'assigned_dev', 'dev_order']].copy()
            original_editable = df[['task_name', 'estimate_md', 'deadline', 'priority', 'assigned_dev', 'dev_order']].copy()

            # VALIDATION: Clean up invalid manually-added rows
            validation_warnings = []
            
            # Validate task names (must not be empty)
            invalid_names = editable_data['task_name'].isna() | (editable_data['task_name'].astype(str).str.strip() == '')
            if invalid_names.any():
                num_invalid = invalid_names.sum()
                validation_warnings.append(f"Removed {num_invalid} task(s) with empty names")
                editable_data = editable_data[~invalid_names].reset_index(drop=True)
            
            # Validate estimates (must be > 0)
            invalid_estimates = editable_data['estimate_md'].isna() | (editable_data['estimate_md'] <= 0)
            if invalid_estimates.any():
                num_invalid = invalid_estimates.sum()
                validation_warnings.append(f"Removed {num_invalid} task(s) with invalid estimates (must be > 0)")
                editable_data = editable_data[~invalid_estimates].reset_index(drop=True)
            
            # Validate deadlines (must not be empty)
            invalid_deadlines = editable_data['deadline'].isna()
            if invalid_deadlines.any():
                num_invalid = invalid_deadlines.sum()
                validation_warnings.append(f"Removed {num_invalid} task(s) with missing deadlines")
                editable_data = editable_data[~invalid_deadlines].reset_index(drop=True)
            
            # Validate assigned developer (must be in list)
            invalid_devs = ~editable_data['assigned_dev'].isin(developer_names)
            if invalid_devs.any():
                num_invalid = invalid_devs.sum()
                validation_warnings.append(f"Reassigned {num_invalid} task(s) with invalid developers to {developer_names[0]}")
                editable_data.loc[invalid_devs, 'assigned_dev'] = developer_names[0]
            
            # Validate order (must be >= 1)
            invalid_orders = editable_data['dev_order'].isna() | (editable_data['dev_order'] < 1)
            if invalid_orders.any():
                num_invalid = invalid_orders.sum()
                validation_warnings.append(f"Fixed {num_invalid} task(s) with invalid order (set to 1)")
                editable_data.loc[invalid_orders, 'dev_order'] = 1
            
            # Show all validation warnings
            if validation_warnings:
                st.warning("⚠️ **Data Validation:**\n- " + "\n- ".join(validation_warnings))

            # Normalize for comparison (deadline can be date vs datetime from editor)
            def _normalize_for_compare(d):
                d = d.copy()
                d['deadline'] = pd.to_datetime(d['deadline']).dt.normalize()
                return d.reset_index(drop=True)

            data_changed = (
                len(editable_data) != len(original_editable)
                or not _normalize_for_compare(editable_data).equals(_normalize_for_compare(original_editable))
            )
            
            # Always update session state with current editable data
            st.session_state.manual_tasks_df = editable_data
            
            if data_changed:
                st.rerun()  # Re-run so simulate_schedule recalculates with new data

            # Check for duplicate orders within same developer
            duplicate_orders = editable_data.groupby('assigned_dev')['dev_order'].apply(
                lambda x: x.duplicated().any()
            )
            
            if duplicate_orders.any():
                problem_devs = duplicate_orders[duplicate_orders].index.tolist()
                st.info(f"ℹ️ **Duplicate order numbers detected** for: {', '.join(problem_devs)}. Tasks with the same order execute by: deadline (earlier first) → task name (alphabetical).")
        st.info("Release days are Mondays and Thursdays. So possible deadlines should be Feb 19, Feb 23, Feb 26, Mar 2, Mar 5, Mar 9, Mar 12, Mar 16, Mar 19, Mar 23, Mar 26, Mar 30, Apr 2, Apr 6, Apr 9")
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # ================================================================
        # TIMELINE VISUALIZATION
        # ================================================================
        
        st.header("🏃 Developer Lanes Timeline")
        
        scheduling_mode = st.radio(
            "**Scheduling mode** — How tasks are assigned to developers:",
            options=['Manual', 'Solver: Minimize Total Days Late', 'Solver: Minimize Maximum Delay'],
            index=['Manual', 'Solver: Minimize Total Days Late', 'Solver: Minimize Maximum Delay'].index(st.session_state.scheduling_mode),
            horizontal=True,
            help="""
            **Manual**: Use assignments from CSV or your edits  
            **Minimize Total Days Late**: Optimize to reduce sum of all delays  
            **Minimize Maximum Delay**: Optimize to reduce the worst-case delay (bottleneck)
            """
        )
        if scheduling_mode != st.session_state.scheduling_mode:
            st.session_state.scheduling_mode = scheduling_mode
            st.rerun()
        
        if is_solver_mode:
            st.info(f"🤖 **Active:** {st.session_state.scheduling_mode} — Assignments and order are optimized. Switch to Manual to use your plan.")
        else:
            st.info(f"✋ **Active:** Manual — Using your assignments. Use solver modes above for optimized alternatives.")
        
        # Prepare data for Gantt chart (include priority from df for coloring)
        priority_by_task = df.set_index('task_name')['priority'].astype(str).str.strip() if 'priority' in df.columns else pd.Series(dtype=object)
        gantt_data = []
        for _, task in sim_results.iterrows():
            tn = task['task_name']
            gantt_data.append({
                'Task': tn,
                'Developer': task['assigned_dev'],
                'Start': task['start_date'],
                'Finish': task['finish_date'],
                'Status': task['status'],
                'Priority': priority_by_task.get(tn, 'Medium'),
                'Days Late': task['days_late'] if task['is_late'] else 0,
                'Estimate (MD)': task['estimate_md']
            })
        
        gantt_df = pd.DataFrame(gantt_data)
        
        # Create timeline chart colored by priority
        _priority_colors = {'Highest': '#4A148C', 'High': '#7B1FA2', 'Medium': '#BA68C8', 'Low': '#F3E5F5'}
        fig = px.timeline(
            gantt_df,
            x_start='Start',
            x_end='Finish',
            y='Developer',
            color='Priority',
            text=gantt_df['Task'].str[:8],
            hover_data={'Task': True, 'Estimate (MD)': True, 'Days Late': True, 'Start': False, 'Finish': True, 'Status': True, 'Priority': False, 'Developer': False},
            color_discrete_map=_priority_colors,
            category_orders={'Priority': ['Highest', 'High', 'Medium', 'Low']},
            title='Task Execution Timeline by Developer (colored by Priority)'
        )
        
        # Update layout
        fig.update_layout(
            height=max(400, len(developer_names) * 80),
            xaxis_title='Date',
            yaxis_title='Developer',
            showlegend=True,
            hovermode='closest'
        )
        
        # Update text position and style
        fig.update_traces(
            textposition='inside',
            textfont_size=10,
            insidetextanchor='middle'
        )
        
        fig.update_yaxes(categoryorder='category descending')
        
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # ================================================================
        # FEASIBILITY KPIs
        # ================================================================
        # Calculate KPIs using sim_results
        num_late = sim_results['is_late'].sum()
        total_days_late = sim_results['days_late'].sum()
        max_days_late = sim_results['days_late'].max()
        all_work_finish = sim_results['finish_date'].max()
        total_tasks = len(sim_results)
        on_time_tasks = total_tasks - num_late
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Late Tasks",
                f"{num_late} / {total_tasks}",
                delta=f"{num_late}" if num_late > 0 else "✓ All on time",
                delta_color="inverse"
            )
        
        with col2:
            st.metric(
                "Total Days Late",
                f"{total_days_late} days",
                delta="Minimized" if is_solver_mode and st.session_state.scheduling_mode == 'Solver: Minimize Total Days Late' else None
            )
        
        with col3:
            st.metric(
                "Maximum Delay",
                f"{max_days_late} days",
                delta="Minimized" if is_solver_mode and st.session_state.scheduling_mode == 'Solver: Minimize Maximum Delay' else ("Critical" if max_days_late > 7 else ("Warning" if max_days_late > 0 else "OK")),
                delta_color="off" if max_days_late > 0 else "normal"
            )
        
        with col4:
            st.metric(
                "All Work Completes",
                all_work_finish.strftime('%Y-%m-%d'),
                delta=f"{(all_work_finish - datetime.now()).days} days from now",
                delta_color="off"
            )
        
        with col5:
            success_rate = (on_time_tasks / total_tasks * 100) if total_tasks > 0 else 0
            st.metric(
                "Success Rate",
                f"{success_rate:.1f}%",
                delta="Feasible" if success_rate == 100 else "Infeasible",
                delta_color="normal" if success_rate == 100 else "inverse"
            )
        
        # Feasibility assessment
        if num_late == 0:
            st.success("✅ **FEASIBLE**: All tasks can be completed by their deadlines with the current plan.")
        else:
            st.error(f"❌ **INFEASIBLE**: {num_late} task(s) will miss their deadlines. Consider: reducing scope, extending deadlines, or increasing capacity.")
        
        # Priority × Effort matrix (with jitter so overlapping tasks are visible)
        st.markdown("<hr>", unsafe_allow_html=True)
        _matrix_df = df[['task_name', 'estimate_md', 'priority']].copy()
        _matrix_df = _matrix_df.merge(sim_results[['task_name', 'status']], on='task_name', how='left')
        _matrix_df['status'] = _matrix_df['status'].fillna('On-time 🟢')
        _priority_order = ['Highest', 'High', 'Medium', 'Low']
        _matrix_df['priority'] = pd.Categorical(
            _matrix_df['priority'].astype(str).str.strip(),
            categories=_priority_order,
            ordered=True
        )
        _matrix_df = _matrix_df.sort_values('priority')
        # Jitter x so tasks with same (priority, estimate_md) don't overlap
        _x_range = _matrix_df['estimate_md'].max() - _matrix_df['estimate_md'].min()
        _jitter_scale = (_x_range * 0.03) if _x_range > 0 else 0.5
        np.random.seed(42)
        _matrix_df['estimate_md_jitter'] = _matrix_df['estimate_md'] + np.random.uniform(-_jitter_scale, _jitter_scale, size=len(_matrix_df))
        fig_matrix = px.scatter(
            _matrix_df,
            x='estimate_md_jitter',
            y='priority',
            color='status',
            hover_data={'task_name': True, 'estimate_md': True, 'priority': True, 'estimate_md_jitter': False},
            title='Tasks by Priority (y) and Effort in person-days (x)',
            color_discrete_map={'On-time 🟢': '#00E676', 'Late 🔴': '#FF1744'},
            category_orders={'priority': _priority_order},
            size=[18] * len(_matrix_df),  # Make all dots bigger by setting a fixed large size
            size_max=18
        )
        fig_matrix.update_layout(
            xaxis_title='Effort (estimate_md)',
            yaxis_title='Priority',
        )
        fig_matrix.update_traces(marker=dict(sizeref=2.0, sizemode='diameter'))  # Fine-tune marker size if needed
        fig_matrix.update_xaxes(
            range=[_matrix_df['estimate_md'].min() - _jitter_scale * 2, _matrix_df['estimate_md'].max() + _jitter_scale * 2],
        )
        st.plotly_chart(fig_matrix, use_container_width=True)
        
        # Summary statistics
        st.markdown("<hr>", unsafe_allow_html=True)
        st.header("📈 Capacity Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Workload Distribution")
            workload_by_dev = sim_results.groupby('assigned_dev')['estimate_md'].sum().reset_index()
            workload_by_dev.columns = ['Developer', 'Total MD']
            
            fig_workload = px.bar(
                workload_by_dev,
                x='Developer',
                y='Total MD',
                title='Total Person-Days by Developer',
                color='Total MD',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_workload, use_container_width=True)
        
        with col2:
            st.subheader("Task Status Breakdown")
            status_counts = sim_results['status'].value_counts().reset_index()
            status_counts.columns = ['Status', 'Count']
            
            fig_status = px.pie(
                status_counts,
                values='Count',
                names='Status',
                title='Tasks by Status',
                color='Status',
                color_discrete_map={'On-time 🟢': '#00E676', 'Late 🔴': '#FF1744'}
            )
            st.plotly_chart(fig_status, use_container_width=True)
    else:
        st.error("❌ Simulation failed. Please check your data for errors.")

else:
    # Welcome screen
    st.info("👈 Upload a task CSV file or generate sample data using the sidebar to begin.")
    
    st.markdown("""
    ### Getting Started
    
    1. **Configure your team** in the sidebar (number of developers, effective hours)
    2. **Upload your task data** as a CSV file with required columns
    3. **Choose a scheduling mode**: Manual or one of the solver options
    4. **Analyze feasibility** using KPIs, detailed tables, and timeline visualization
    
    ### Scheduling Modes
    
    - **Manual**: Use the assignments from your CSV or make edits in the table
    - **Solver: Minimize Total Days Late**: Automatically optimize to reduce total lateness
    - **Solver: Minimize Maximum Delay**: Automatically optimize to nimimize delay per task
    

    ### Sample CSV Format

    **Required CSV columns:**
    - `task_name` (string)
    - `estimate_md` (float, person-days, must be > 0)
    - `deadline` (date, YYYY-MM-DD)

    **Optional columns:**
    - `priority` (text: Highest, High, Medium, Low)
    - `assigned_dev` (string, developer name)
    - `dev_order` (integer, must be > 0)

    ```csv
    task_name,estimate_md,deadline,priority,assigned_dev,dev_order
    Task 1,5.0,2026-02-15,Highest,Ahmet,1
    Task 2,3.5,2026-02-20,High,Mehmet,1
    Task 3,8.0,2026-03-01,Medium,Ayşe,1
    ```

    
    
    ### Important Notes
    
    - All estimates must be greater than 0
    - Deadlines must be in YYYY-MM-DD format
    - Solver modes generate optimized schedules automatically
    - Switch back to Manual mode to restore your original plan
    """)
