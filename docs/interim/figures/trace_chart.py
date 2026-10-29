"""Stage times of the clean 21 Oct 2026 run, from the trace file ecde8905-20261021T044917Z-ac926b.jsonl."""
import json, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for f in fm.findSystemFonts():
    if 'Inter' in f: fm.fontManager.addfont(f)
plt.rcParams.update({'font.family': 'Inter', 'font.size': 11, 'axes.edgecolor': '#9a9893',
                     'xtick.color': '#52514e', 'ytick.color': '#1f1f1f'})
ev = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
# group the events into the steps the loop takes
steps, cur = [], None
names = {'grounding': 'Grounding', 'planning': 'Plan', 'execution': None}
perceive = 0.0
skills = iter(['Pick and check', 'Place and check'])
for e in ev:
    if e['type'] != 'event': continue
    st, d = e['stage'], e['duration_s']
    if st in ('capture', 'detection', 'depth', 'transform'):
        perceive += d; continue
    if perceive:
        steps.append(('Perceive', perceive)); perceive = 0.0
    if st == 'grounding': steps.append(('Grounding', d))
    elif st == 'planning': steps.append(('Plan', d))
    elif st == 'execution': steps.append((next(skills), d))
total = [e for e in ev if e['type'] == 'run_end'][0]['total_s']
labels = [f'{i + 1}. {n}' for i, (n, _) in enumerate(steps)]; labels[-1] = labels[-1] + ', done'
vals = [v for _, v in steps]
cols = ['#e2874a' if 'check' in n else '#7d62d6' if n in ('Plan', 'Grounding') else '#4a86d4' for n, _ in steps]
fig, ax = plt.subplots(figsize=(7.4, 3.6), dpi=240)
y = range(len(vals))[::-1]
ax.barh(list(y), vals, color=cols, height=0.62, zorder=3)
for yi, v in zip(y, vals):
    ax.text(v + 0.15, yi, f'{v:.2f} s', va='center', fontsize=10, color='#1f1f1f')
ax.set_yticks(list(y), labels)
ax.set_xlabel('Time (s)', color='#1f1f1f')
ax.set_xlim(0, max(vals) * 1.18)
ax.grid(axis='x', color='#e6e4df', zorder=0)
ax.spines[['top', 'right', 'left']].set_visible(False)
ax.tick_params(axis='y', length=0)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color='#7d62d6', label='LLM'), Patch(color='#4a86d4', label='Perception'),
                   Patch(color='#e2874a', label='Arm motion and check')],
          frameon=False, loc='upper right', fontsize=10)
ax.set_title(f'Total {total:.1f} s, of which arm motion {sum(v for n, v in steps if "check" in n):.1f} s',
             fontsize=11, color='#52514e', loc='left')
fig.tight_layout(); fig.savefig(sys.argv[2], facecolor='white')
print(steps, total)
