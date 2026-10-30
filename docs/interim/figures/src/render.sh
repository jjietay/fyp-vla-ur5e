#!/usr/bin/env bash
# Renders each figure's HTML to the PNG the interim report uses.
#   ./render.sh            all figures
#   ./render.sh figure_2   one figure
#   OUT=/tmp/x ./render.sh writes there instead of next to the report figures
# Each HTML file sets its canvas size in a <meta name="canvas"> tag, and the
# scale gives the same pixel size as the PNG in the report, so a re-render
# drops into the .docx without resizing.
set -euo pipefail
cd "$(dirname "$0")"

render() {
  local html=$1 out=$2
  read -r w h scale < <(sed -n 's/.*<meta name="canvas" content="\([0-9]*\) \([0-9]*\) \([0-9.]*\)".*/\1 \2 \3/p' "$html")
  google-chrome --headless=new --disable-gpu --hide-scrollbars --no-sandbox \
    --force-device-scale-factor="$scale" --window-size="$w,$h" \
    --virtual-time-budget=2000 --screenshot="$out" "file://$PWD/$html" 2>/dev/null
  echo "$out"
}

dir=${OUT:-..}
for f in figure_*.html; do
  name=${f%.html}
  [[ $# -gt 0 && $name != "$1"* ]] && continue
  case $name in
    figure_1*) out=$dir/Figure_1_full_system.png ;;
    figure_2*) out=$dir/Figure_2_architecture_A.png ;;
    figure_3*) out=$dir/Figure_3_architecture_B.png ;;
    figure_9*) out=$dir/Figure_9_schedule.png ;;
    figure_code_boundary) out=$dir/Figure_code_boundary.png ;;
    figure_calibration) out=$dir/Figure_calibration.png ;;
    *) out=$dir/$name.png ;;
  esac
  render "$f" "$out"
done
