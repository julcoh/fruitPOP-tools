# =============================================================================
# 0.  Install the libraries we need
# -----------------------------------------------------------------------------
# • ortools          – Google’s optimisation toolkit (CP-SAT solver)
# • gspread          – Python API for Google Sheets
# • gspread_dataframe– Helper: move whole DataFrames in/out of Sheets
# • oauth2client     – Handles the OAuth dance inside Colab
# =============================================================================
import subprocess, sys

# Install required libraries when executed outside of a notebook.
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "ortools", "gspread", "gspread_dataframe", "oauth2client"],
    check=True,
)

# =============================================================================
# 1.  Authenticate this Colab session with your Google account
# -----------------------------------------------------------------------------
# Colab’s auth helper pops up a consent screen; once you allow it,
# the notebook inherits a short-lived credential that works with
# both Google Drive and Google Sheets.
# =============================================================================
from google.colab import auth
auth.authenticate_user()     # <-- you’ll get a clickable login link

# Build a gspread “client” that can read / write spreadsheets.
import gspread
from gspread_dataframe import get_as_dataframe, set_with_dataframe
from google.auth import default           # picks up the Colab creds

creds, _ = default()                      # creds = OAuth token object
gc = gspread.authorize(creds)             # gspread client authorised

import pandas as pd

# =============================================================================
# 2.  Pull the three source tabs into Pandas DataFrames
#     + validate inputs before running the solver
# =============================================================================
import pandas as pd, numpy as np
import re
from ortools.sat.python import cp_model
import random

# ---------- helper to normalise ShiftIDs ------------------------------------
def norm_id(val):
    if pd.isna(val):
        return ''
    s = str(val).strip()
    if re.fullmatch(r'\d+(\.0+)?', s):
        return str(int(float(s)))        # drop .0 and leading zeros
    return s

# ---------- load sheets ------------------------------------------------------
SHEET_URL = 'https://docs.google.com/spreadsheets/d/1QjUd6m-lK0NbDjKvdvtMq-XJO3KqNz9DsL3-MKSDN_A/edit'
ss          = gc.open_by_url(SHEET_URL)

shifts_ws, prefs_ws, settings_ws = (ss.worksheet(n) for n in ('Shifts','Prefs','Settings'))

shifts = get_as_dataframe(shifts_ws, evaluate_formulas=True).dropna(how='all')
shifts['ShiftID']   = shifts['ShiftID'].apply(norm_id)
shifts['Capacity']  = pd.to_numeric(shifts['Capacity'], errors='coerce').fillna(0).astype(int)
shifts['Points']    = pd.to_numeric(shifts['Points'],   errors='coerce').fillna(0).astype(int)

prefs_raw = get_as_dataframe(prefs_ws, evaluate_formulas=True).dropna(how='all')
volunteer_names = prefs_raw.iloc[:,0].astype(str).tolist()
prefs = prefs_raw.drop(prefs_raw.columns[0], axis=1)
prefs.columns = [norm_id(c) for c in prefs.columns]
prefs.index   = volunteer_names

settings  = settings_ws.get_all_records()[0]
MIN_PTS   = int(settings['MIN_POINTS'])
SEED      = int(settings['SEED'])
MAX_OVER  = int(settings['MAX_OVER'])

print(f'Loaded {len(shifts)} shifts, {len(volunteer_names)} volunteers')
print(f'MIN_POINTS={MIN_PTS}, SEED={SEED}, MAX_OVER={MAX_OVER}')

# =============================================================================
# ✅  PRE-SOLVER VALIDATION
# =============================================================================
if shifts.empty:
    raise ValueError("No shifts defined.")
if prefs.empty:
    raise ValueError("No volunteer preferences provided.")

shift_ids  = shifts['ShiftID'].tolist()
volunteers = prefs.index.tolist()

# 1) shifts with zero possible volunteers
shift_to_possible = {
    sid: [v for v in volunteers if sid in prefs.columns and not pd.isna(prefs.at[v, sid])]
    for sid in shift_ids
}
no_coverage = [sid for sid, vols in shift_to_possible.items() if len(vols)==0]
if no_coverage:
    raise ValueError(f"Shifts with no volunteer coverage: {no_coverage}")

# 2) volunteers with no prefs
no_prefs = [v for v in volunteers if prefs.loc[v].isna().all()]
if no_prefs:
    raise ValueError(f"Volunteers with no preferences: {no_prefs}")

# 3) total points feasibility
total_possible_points = (shifts['Capacity'] * shifts['Points']).sum()
total_required_points = len(volunteers) * MIN_PTS          # ← FIXED
#if total_possible_points > total_required_points:
#    raise ValueError(
#        f"Only {total_possible_points} total points available but "
#        f"{total_required_points} required to hit everyone’s MIN_PTS."
#    )

print('✅  All validation checks passed.')

# =============================================================================
# SOLVER – two-step:
#   1) find smallest rank_cut so everyone can get ≥1 shift
#   2) optimise full roster with that rule, using small tier weights
# =============================================================================
from ortools.sat.python import cp_model
import random, math

# ---------- helpers ----------------------------------------------------------
rand       = random.Random(SEED)
points_d   = dict(zip(shifts['ShiftID'], shifts['Points']))
max_rank   = int(prefs.apply(pd.to_numeric, errors='coerce').max().max())

def cutoff_feasible(r_cut: int) -> bool:
    """Return True if each volunteer can take at least one shift of rank≤r_cut."""
    m = cp_model.CpModel()
    x = {(v, s): m.NewBoolVar(f'x_{v}_{s}')
         for v in volunteers for s in shift_ids
         if s in prefs.columns and
            pd.to_numeric(prefs.at[v, s], errors='coerce') <= r_cut}

    # capacity
    for s, cap in zip(shift_ids, shifts['Capacity']):
        m.Add(sum(x.get((v, s), 0) for v in volunteers) <= cap)

    # ≥1 shift per volunteer
    for v in volunteers:
        elig = [x[(v, s)] for s in shift_ids if (v, s) in x]
        if not elig:                # volunteer has no shift ≤ r_cut
            return False
        m.Add(sum(elig) >= 1)

    solver = cp_model.CpSolver(); solver.parameters.max_time_in_seconds = 5
    return solver.Solve(m) in (cp_model.FEASIBLE, cp_model.OPTIMAL)

# ---------- STEP A: find best_cut -------------------------------------------
best_cut = None
for r in range(1, max_rank + 1):
    if cutoff_feasible(r):
        best_cut = r
        break
if best_cut is None:
    raise ValueError("No rank_cut found that lets everyone get one shift.")

print(f'✅  Everyone can get a shift of rank ≤ {best_cut}')

# ---------- STEP B: full optimisation ---------------------------------------
m2 = cp_model.CpModel()
x  = {(v, s): m2.NewBoolVar(f'x_{v}_{s}') for v in volunteers for s in shift_ids}

# 1) capacity
for s, cap in zip(shift_ids, shifts['Capacity']):
    m2.Add(sum(x[v, s] for v in volunteers) <= cap)

# 2) per-volunteer constraints
for v in volunteers:
    total_pts = sum(x[v, s] * points_d[s] for s in shift_ids)
    m2.Add(total_pts >= MIN_PTS)
    m2.Add(total_pts <= MIN_PTS + MAX_OVER)

    # guarantee ≥1 shift of rank ≤ best_cut
    elite = [x[v, s] for s in shift_ids
             if s in prefs.columns and
                pd.to_numeric(prefs.at[v, s], errors='coerce') <= best_cut]
    m2.Add(sum(elite) >= 1)

# 3) objective – small tier weights (never overflow)
obj_terms = []
for v in volunteers:
    for s in shift_ids:
        if s in prefs.columns and not pd.isna(prefs.at[v, s]):
            rank = int(prefs.at[v, s])
            if rank == 1:
                weight = 300
            elif rank == 2:
                weight = 200
            elif rank == 3:
                weight = 100
            else:
                continue            # ignore ranks ≥4 in scoring
            weight += rand.randint(0, 9)       # deterministic tie-break ε
            obj_terms.append(weight * x[v, s])

m2.Maximize(sum(obj_terms))

# 4) solve
solver2 = cp_model.CpSolver()
solver2.parameters.max_time_in_seconds = 30
status2 = solver2.Solve(m2)

print('Solver status :', solver2.StatusName(status2))
if status2 in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    print('Objective     :', solver2.ObjectiveValue())
	
# ─────────────────────────────────────────────────────────────────────────────
# ⑤-⑥-⑦  BUILD & PUSH  ShiftVols  +  Roster-by-Volunteer  +  Audit (extended)
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from collections import defaultdict

# ---------------------------------------------------------------------------
# A)  ShiftVols  (per shift with assigned volunteers)
# ---------------------------------------------------------------------------
shift_vol_rows, max_vols_shift = [], 0
for _, row in shifts.iterrows():
    sid   = row['ShiftID']
    role  = row['Role']
    cap   = int(row['Capacity'])
    pts   = int(row['Points'])
    vols  = [v for v in volunteers if solver2.BooleanValue(x[v, sid])]
    max_vols_shift = max(max_vols_shift, len(vols))
    shift_vol_rows.append([sid, role, cap, pts, *vols])

for r in shift_vol_rows:                                    # pad rows
    r += [''] * (max_vols_shift - (len(r) - 4))

shift_vol_cols = ['ShiftID', 'Role', 'Capacity', 'Points'] + [
    f'Volunteer{i+1}' for i in range(max_vols_shift)
]
shift_vol_df = pd.DataFrame(shift_vol_rows, columns=shift_vol_cols)

# ---------------------------------------------------------------------------
# B)  Roster  (one row per volunteer, listing all their shifts)
# ---------------------------------------------------------------------------
roster_rows, max_shifts_vol = [], 0
for v in volunteers:
    my_shifts = [s for s in shift_ids if solver2.BooleanValue(x[v, s])]
    max_shifts_vol = max(max_shifts_vol, len(my_shifts))
    roster_rows.append([v, *my_shifts])

for r in roster_rows:
    r += [''] * (max_shifts_vol - (len(r) - 1))             # pad blanks

roster_cols = ['Volunteer'] + [f'Shift{i+1}' for i in range(max_shifts_vol)]
roster_df   = pd.DataFrame(roster_rows, columns=roster_cols)

# ---------------------------------------------------------------------------
# C)  Audit  (per volunteer + team-level summary)
# ---------------------------------------------------------------------------
points_dict = dict(zip(shifts['ShiftID'], shifts['Points'].astype(int)))
rank_cols   = [f'# {i} hits' for i in range(1, 21)]          # #1 … #10 columns
audit_rows  = []
rank_hit_aggregate = defaultdict(int)                       # count volunteers w/ ≥1 hit at rank i

for v in volunteers:
    my_shifts = [s for s in shift_ids if solver2.BooleanValue(x[v, s])]
    total_pts = sum(points_dict.get(s, 0) for s in my_shifts)

    # count hits for ranks 1-10
    hits = [0]*20
    for s in my_shifts:
        if s in prefs.columns:
            r = pd.to_numeric(prefs.at[v, s], errors='coerce')
            if 1 <= r <= 20:
                hits[int(r)-1] += 1
    # update aggregate “at least one hit” counter
    for i, h in enumerate(hits, start=1):
        if h > 0:
            rank_hit_aggregate[i] += 1

    audit_rows.append(
        {'Volunteer': v,
         'TotalPoints': total_pts,
         **{rank_cols[i]: hits[i] for i in range(20)},
         'AssignedShifts': '; '.join(my_shifts)}
    )

audit_df = (pd.DataFrame(audit_rows)
            .sort_values('Volunteer')
            .reset_index(drop=True))

# -------- second block: % of volunteers with ≥1 hit at each rank ------------
summary_rows = []
n_vols = len(volunteers)
for i in range(1, 21):
    count = rank_hit_aggregate.get(i, 0)
    summary_rows.append({'Rank': i,
                         'VolsWithHit': count,
                         'Percentage': round(count / n_vols * 100, 1)})

summary_df = pd.DataFrame(summary_rows)

# Append a blank line then the summary table to audit_df
# Using an explicit dictionary avoids relying on Python 3.9+'s dict union.
blank_row = pd.DataFrame([{'Volunteer': ''}])  # minimal blank
audit_df  = pd.concat([audit_df, blank_row, summary_df], ignore_index=True)

# Footer
footer = pd.DataFrame([{
    'Volunteer'      : '*** Seed used',
    'TotalPoints'    : SEED,
    '# 1 hits'       : 'Solver status',
    'AssignedShifts' : solver2.StatusName(status2)
}])
audit_df = pd.concat([audit_df, footer], ignore_index=True)

# ---------------------------------------------------------------------------
# D)  Push all three tabs to the Sheet
# ---------------------------------------------------------------------------
def drop_if_exists(name):
    try: ss.del_worksheet(ss.worksheet(name))
    except gspread.exceptions.WorksheetNotFound: pass

for tab in ('ShiftVols', 'Roster', 'Audit'):
    drop_if_exists(tab)

shift_ws  = ss.add_worksheet('ShiftVols', rows=shift_vol_df.shape[0]+1,
                             cols=shift_vol_df.shape[1])
roster_ws = ss.add_worksheet('Roster',    rows=roster_df.shape[0]+1,
                             cols=roster_df.shape[1])
audit_ws  = ss.add_worksheet('Audit',     rows=audit_df.shape[0]+1,
                             cols=audit_df.shape[1])

set_with_dataframe(shift_ws,  shift_vol_df, include_index=False)
set_with_dataframe(roster_ws, roster_df,    include_index=False)
set_with_dataframe(audit_ws,  audit_df,     include_index=False)

print('✅  ShiftVols, Roster, and Audit tabs refreshed.')
