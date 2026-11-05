"""Clean the OSRM district workbook for PULSE.

1. Re-place 3 mis-geocoded districts and add 5 missing ones. Their distances are
   estimated from the nearest correctly placed district's routed row:
       d(X, j) = H(X, j) * d(N, j) / H(N, j)
   i.e. X inherits N's real road-detour factor to every destination.
2. Blank the diagonal (PULSE fills it with U[10, 20] km).
3. Blank island <-> mainland cells (no road link). Andaman & Nicobar districts stay
   connected to each other; Lakshadweep connects to nothing but itself.
4. Add a `notes` sheet documenting all of this.

Run from the project root:  python data_prep/fix_excel.py data/districts.xlsx
Inputs (in data_prep/): districts_osrm_raw.xlsx (output of webScraper.ipynb),
districts_coordinates.csv, district_id_map.csv. Optional: a district-boundary GeoJSON
at data_prep/districts.geojson (e.g. github.com/datta07/INDIAN-SHAPEFILES) to report which
district polygon each re-placed point falls in; needs `pip install shapely`.
"""
import json, os, sys
import numpy as np, pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "")
OUT = sys.argv[1] if len(sys.argv) > 1 else "districts_clean.xlsx"
R_EARTH = 6371.0088

xd = pd.read_excel(SRC + "districts_osrm_raw.xlsx", sheet_name="districts")
xm = pd.read_excel(SRC + "districts_osrm_raw.xlsx", sheet_name="distances", index_col=0)
coords = pd.read_csv(SRC + "districts_coordinates.csv")
idmap = pd.read_csv(SRC + "district_id_map.csv")
cm = coords.set_index("district_id").loc[idmap.old_id].reset_index()
xd["lat"] = cm.latitude.to_numpy(); xd["lon"] = cm.longitude.to_numpy()
assert list(xm.index) == list(xd.district_id) == list(xm.columns)
M = xm.to_numpy(float)

# HQ coordinates (lat, lon)
FIXES = {"Anakapalli": (17.6896, 83.0024), "Ananthapuramu": (14.6819, 77.6006), "Bagpat": (28.9447, 77.2183)}
ADD = [  # name, state, lat, lon
    ("Manendragarh-Chirmiri-Bharatpur", "Chhattisgarh", 23.2050, 82.2050),
    ("Shahid Bhagat Singh Nagar", "Punjab", 31.1250, 76.1160),
    ("Dadra and Nagar Haveli", "Dadra and Nagar Haveli and Daman and Diu", 20.2740, 73.0090),
    ("Daman", "Dadra and Nagar Haveli and Daman and Diu", 20.4140, 72.8320),
    ("Diu", "Dadra and Nagar Haveli and Daman and Diu", 20.7140, 70.9820),
]


def hav(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


ISLAND = xd.state.str.contains("Andaman|Lakshadweep").to_numpy()
log = []

# ---- check new points against district boundaries
GEO = SRC + "districts.geojson"
if os.path.exists(GEO):
    from shapely.geometry import shape, Point
    g = json.load(open(GEO))
    polys = [(f["properties"]["district"], f["properties"]["state"], shape(f["geometry"]).buffer(0)) for f in g["features"]]
    def where(lat, lon):
        p = Point(lon, lat)
        return next((f"{d} ({s})" for d, s, poly in polys if poly.contains(p)), "none")
else:
    def where(lat, lon):
        return "(boundary file not available)"


def estimate_row(lat, lon, exclude):
    """Distances from (lat, lon) to every current district, via nearest routed neighbour."""
    h_all = hav(lat, lon, xd.lat.to_numpy(), xd.lon.to_numpy())
    cand = np.where(~exclude & ~ISLAND)[0]
    n = cand[np.argmin(h_all[cand])]
    hn = hav(xd.lat[n], xd.lon[n], xd.lat.to_numpy(), xd.lon.to_numpy())
    row_n = M[n].copy()
    ok = (hn > 30) & np.isfinite(row_n) & ~ISLAND
    med = float(np.median(row_n[ok] / hn[ok]))
    ratio = np.where(hn > 30, row_n / np.where(hn > 0, hn, 1), med)
    ratio = np.clip(ratio, 1.0, None)
    est = np.round(h_all * ratio, 3)
    est = np.maximum(est, np.round(h_all * 1.0, 3))
    return est, n, med


# ---- 1a. re-place mis-geocoded districts
bad = np.zeros(len(xd), bool)
for name in FIXES: bad |= (xd.name == name).to_numpy()
for name, (la, lo) in FIXES.items():
    i = int(np.where(xd.name == name)[0][0])
    old = (xd.lat[i], xd.lon[i])
    xd.loc[i, ["lat", "lon"]] = [la, lo]
    est, n, med = estimate_row(la, lo, bad)
    M[i, :] = est; M[:, i] = est
    log.append(f"{xd.district_id[i]} {name}: moved from ({old[0]:.3f}, {old[1]:.3f}) to ({la}, {lo}), now inside {where(la, lo)}; "
               f"distances estimated from {xd.name[n]} ({xd.district_id[n]}), median detour x{med:.2f}")

# ---- 1b. add missing districts
start = len(xd)
for k, (name, state, la, lo) in enumerate(ADD):
    new_id = f"D{start + k + 1:03d}"
    est, n, med = estimate_row(la, lo, bad)
    # grow matrix
    M = np.pad(M, ((0, 1), (0, 1)), constant_values=np.nan)
    M[-1, :-1] = est; M[:-1, -1] = est
    xd.loc[len(xd)] = {"district_id": new_id, "name": name, "state": state, "lat": la, "lon": lo}
    ISLAND = np.append(ISLAND, False); bad = np.append(bad, True)
    log.append(f"{new_id} {name} ({state}): added at ({la}, {lo}), inside {where(la, lo)}; "
               f"distances estimated from {xd.name[n]} ({xd.district_id[n]}), median detour x{med:.2f}")
# distances among the added districts themselves (and to fixed ones): use straight line x neighbour detour already in row;
# make symmetric by averaging the two estimates
M = np.where(np.isnan(M), M.T, M)
M = (M + M.T) / 2

# ---- 2. diagonal blank
np.fill_diagonal(M, np.nan)

# ---- 3. islands
anda = xd.state.str.contains("Andaman").to_numpy()
laks = xd.state.str.contains("Lakshadweep").to_numpy()
mask = (anda[:, None] & ~anda[None, :]) | (~anda[:, None] & anda[None, :]) | laks[:, None] | laks[None, :]
np.fill_diagonal(mask, False)
M[mask] = np.nan
n_island = int(np.triu(mask, 1).sum())
log.append(f"Blanked {n_island} island pairs (Andaman & Nicobar <-> mainland, Lakshadweep <-> everything): no road link, PULSE treats blank as unreachable")
log.append("Diagonal left blank: PULSE fills within-district distance with U[10, 20] km")

# ---- checks
n = len(M); off = ~np.eye(n, dtype=bool)
fin = np.isfinite(M)
assert np.allclose(np.where(fin, M, 0), np.where(fin, M, 0).T), "not symmetric"
assert (M[fin & off] > 0).all(), "zero/negative off-diagonal"
H = hav(xd.lat.to_numpy()[:, None], xd.lon.to_numpy()[:, None], xd.lat.to_numpy()[None], xd.lon.to_numpy()[None])
r = M[fin & off & (H > 30)] / H[fin & off & (H > 30)]
assert r.min() >= 0.999, r.min()
print("\n".join(log))
print(f"{n} districts, {int(np.triu(fin & off, 1).sum())} routed/estimated pairs, ratio median {np.median(r):.3f} min {r.min():.3f}")

# ---- write
wb = Workbook()
wb._named_styles["Normal"].font = Font(name="Arial", size=10)
ws = wb.active; ws.title = "districts"
ws.append(["district_id", "name", "state", "latitude", "longitude"])
for r_ in xd.itertuples():
    ws.append([r_.district_id, r_.name, r_.state, round(float(r_.lat), 6), round(float(r_.lon), 6)])
for c in ws[1]: c.font = Font(name="Arial", bold=True)
ws.column_dimensions["B"].width = 32; ws.column_dimensions["C"].width = 40; ws.freeze_panes = "A2"

wd = wb.create_sheet("distances")
ids = list(xd.district_id)
wd.append(["district_id"] + ids)
for lab, row in zip(ids, M.tolist()):
    wd.append([lab] + [None if (v != v) else round(v, 3) for v in row])
for c in wd[1]: c.font = Font(name="Arial", bold=True)
for rr in range(2, wd.max_row + 1): wd.cell(rr, 1).font = Font(name="Arial", bold=True)
wd.freeze_panes = "B2"

wn = wb.create_sheet("notes")
notes = [
    "Road distances in km between district headquarters.",
    "Source: districts scraped from Wikipedia 'List of districts in India', geocoded with Nominatim (OpenStreetMap),",
    "routed with OSRM (project-osrm.org) on OpenStreetMap roads. Pairs OSRM could not route: straight line x 1.3.",
    "Blank diagonal: within-district distance is drawn by PULSE from U[10, 20] km.",
    "Blank off-diagonal cell: no road link (islands); PULSE never matches that pair.",
    "",
    "Cleaning applied after the OSRM run:",
] + log
for line in notes: wn.append([line])
wn.column_dimensions["A"].width = 160
wb.save(OUT)
print("wrote", OUT)
