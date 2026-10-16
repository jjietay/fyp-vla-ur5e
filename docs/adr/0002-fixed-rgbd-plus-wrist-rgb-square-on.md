# A fixed RGB-D plus a wrist RGB camera, the fixed one square on to the table and 0.7 to 0.8 m above it

SmolVLA has no depth input, and A needs depth only from the fixed camera, so the wrist camera is RGB and **nothing may depend on wrist depth**. Two views because the reference SmolVLA setups use overhead plus wrist. The fixed D435i faces the arm along the base x axis at `y = 0`, so range error falls almost entirely on one axis and the reach test can measure it as one constant. 0.7 to 0.8 m is just past the height where short objects stop hiding each other (shadow length `s = d*H/(h-H)`). Tall objects still occlude, so Tier 1 and 2 layouts must keep nothing behind them.

## Consequences

Square on foreshortens the reach axis, which costs B alone, since A uses depth. Accepted because the wrist camera sees approach distance directly. State this bias in the report.
