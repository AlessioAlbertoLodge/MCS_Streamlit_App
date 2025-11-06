# digitize_timeseries.py
# Usage: python digitize_timeseries.py path/to/plot_image.png
# Requires: matplotlib, numpy

import sys, csv
import numpy as np
import matplotlib.pyplot as plt

def ask_float(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Please enter a number (e.g., 0, 12.5, 202.0).")

if len(sys.argv) < 2:
    print("Usage: python digitize_timeseries.py path/to/image")
    sys.exit(1)

img_path = sys.argv[1]
img = plt.imread(img_path)

print("\nINSTRUCTIONS")
print("1) You'll click two points on the X axis you know the time values for (e.g., 0 s and 1800 s).")
print("2) You'll click two points on the Y axis you know the power values for (e.g., 0 kW and 250 kW).")
print("3) You'll then click along the curve (left-click to add points). Press ENTER when done.\n")

fig, ax = plt.subplots()
ax.imshow(img)
ax.set_title("Click TWO known X-axis points (e.g., t=0 and t=end), then press ENTER")
plt.axis('on')
pts = plt.ginput(n=-1, timeout=0)
if len(pts) < 2:
    print("Need two points for X calibration. Exiting.")
    sys.exit(1)
xcal = pts[:2]

# Ask their real values
x0_val = ask_float("Actual time value for first X point: ")
x1_val = ask_float("Actual time value for second X point: ")

# Now Y calibration
ax.set_title("Click TWO known Y-axis points (e.g., P=0 and P=max), then press ENTER")
plt.draw()
pts2 = plt.ginput(n=-1, timeout=0)
if len(pts2) < 2:
    print("Need two points for Y calibration. Exiting.")
    sys.exit(1)
ycal = pts2[:2]

y0_val = ask_float("Actual power value for first Y point: ")
y1_val = ask_float("Actual power value for second Y point: ")

# Curve picking
ax.set_title("Click ALONG the curve (left-click). Press ENTER when finished.")
plt.draw()
curve_pts = plt.ginput(n=-1, timeout=0)
plt.close(fig)

if len(curve_pts) < 2:
    print("You clicked fewer than 2 points for the curve. Exiting.")
    sys.exit(1)

# Convert pixel -> data using linear mapping from the two calibration points
# Note: Matplotlib image coords: x = column (increasing right), y = row (increasing down).
# We'll handle that consistently for linear axes.

# Build mapping for X
(xp0, yp_dummy0), (xp1, yp_dummy1) = xcal
# Protect against division by zero
if xp1 == xp0:
    raise ValueError("X calibration points have the same x pixel. Pick two distinct x positions.")
x_scale = (x1_val - x0_val) / (xp1 - xp0)
x_offset = x0_val - x_scale * xp0

# Build mapping for Y
(xp_dummy0, yp0), (xp_dummy1, yp1) = ycal
if yp1 == yp0:
    raise ValueError("Y calibration points have the same y pixel. Pick two distinct y positions.")
# Because image y increases downward, the mapping inverts sign
y_scale = (y1_val - y0_val) / (yp1 - yp0)
y_offset = y0_val - y_scale * yp0

curve_pts = np.array(curve_pts)
px = curve_pts[:,0]
py = curve_pts[:,1]

tx = x_scale * px + x_offset
py_val = y_scale * py + y_offset

# Sort by time (in case clicking went back and forth)
order = np.argsort(tx)
tx = tx[order]
py_val = py_val[order]

# Optional: deduplicate identical times
mask = np.concatenate(([True], np.diff(tx) != 0))
tx = tx[mask]
py_val = py_val[mask]

out_csv = img_path.rsplit(".", 1)[0] + "_digitized.csv"
with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["time", "power"])
    for t, p in zip(tx, py_val):
        w.writerow([t, p])

print(f"\nSaved: {out_csv}")
print("Done.")
