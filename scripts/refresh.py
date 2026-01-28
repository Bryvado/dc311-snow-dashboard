from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from shapely.geometry import Point, shape
from shapely.strtree import STRtree
import json
import pandas as pd

def clean_str(v):
    if v is None or (isinstance(v, float) and pd.isna(v)) or pd.isna(v):
        return None
    return str(v)

def clean_bool(v):
    if v is None or pd.isna(v):
        return None
    return bool(v)

def clean_float(v):
    if v is None or pd.isna(v):
        return None
    return float(v)

# -----------------
# Config
# -----------------
LAYER_URL = os.getenv(
    "LAYER_URL",
    "https://maps2.dcgis.dc.gov/dcgis/rest/services/DCGIS_APPS/SR_30days_Open/MapServer/17",
)
OUT_DIR = Path(os.getenv("OUT_DIR", "docs/data"))
ANC_PATH = Path(os.getenv("ANC_PATH", "data/anc.geojson"))
WARDS_PATH = Path(os.getenv("WARDS_PATH", "data/wards.geojson"))

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1000"))
LOCAL_TZ = os.getenv("LOCAL_TZ", "America/New_York")


# -----------------
# ArcGIS fetch (paged)
# -----------------
def _arcgis_query_page(result_offset: int) -> Dict[str, Any]:
    query_url = LAYER_URL.rstrip("/") + "/query"
    params = {
        "where": "1=1",
        "outFields": "*",
        "returnGeometry": "true",
        "f": "geojson",
        "resultOffset": str(result_offset),
        "resultRecordCount": str(BATCH_SIZE),
        "orderByFields": "OBJECTID ASC",
        "outSR": "4326",
    }
    r = requests.get(query_url, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def fetch_all_features() -> List[Dict[str, Any]]:
    all_feats: List[Dict[str, Any]] = []
    offset = 0

    while True:
        data = _arcgis_query_page(offset)
        feats = data.get("features", []) or []
        if len(feats) == 0:
            break

        all_feats.extend(feats)

        if len(feats) < BATCH_SIZE:
            break

        offset += BATCH_SIZE

    return all_feats


# -----------------
# Geometry helpers (fast point-in-polygon)
# -----------------
def load_polygons(path: Path, id_field: str) -> Dict[str, Any]:
    gj = json.loads(path.read_text(encoding="utf-8"))
    polys = []
    ids = []
    for ft in gj.get("features", []):
        geom = ft.get("geometry")
        props = ft.get("properties", {}) or {}
        if not geom:
            continue
        polys.append(shape(geom))
        ids.append(str(props.get(id_field)))
    return {"geojson": gj, "polys": polys, "ids": ids}


def assign_ids(points: List[Optional[Point]], polys: List[Any], ids: List[str]) -> List[Optional[str]]:
    """
    Assign polygon id to each point using STRtree + covers().
    Handles STRtree.query returning numpy arrays or geometries across shapely builds.
    """
    tree = STRtree(polys)
    wkb_to_idx = {polys[i].wkb: i for i in range(len(polys))}

    out: List[Optional[str]] = []
    for pt in points:
        if pt is None:
            out.append(None)
            continue

        hits = tree.query(pt)
        if hits is None or len(hits) == 0:
            out.append(None)
            continue

        if isinstance(hits[0], (int, np.integer)):
            cand_idxs = [int(i) for i in hits]
        else:
            cand_idxs = []
            for g in hits:
                idx = wkb_to_idx.get(g.wkb)
                if idx is not None:
                    cand_idxs.append(idx)

        found = None
        for i in cand_idxs:
            if polys[i].covers(pt):
                found = ids[i]
                break

        out.append(found)

    return out


def merge_metrics_into_geojson(gj: Dict[str, Any], key_field: str, metric_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    out = json.loads(json.dumps(gj))  # deep copy
    for ft in out.get("features", []):
        props = ft.get("properties", {}) or {}
        k = props.get(key_field)
        if k is None:
            continue
        add = metric_map.get(str(k))
        if add:
            props.update(add)
        ft["properties"] = props
    return out


# -----------------
# Date parsing + metrics
# -----------------
def parse_dt_series(s: pd.Series) -> pd.Series:
    """
    Handles ISO-like strings and epoch milliseconds.
    Treat naive datetimes as LOCAL_TZ.
    """
    def parse_one(x):
        if pd.isna(x):
            return pd.NaT
        if isinstance(x, (int, float)) and x > 1e11:
            return pd.to_datetime(int(x), unit="ms", utc=True).tz_convert(LOCAL_TZ)
        dt = pd.to_datetime(x, errors="coerce")
        if pd.isna(dt):
            return pd.NaT
        if getattr(dt, "tzinfo", None) is None:
            return dt.tz_localize(LOCAL_TZ)
        return dt.tz_convert(LOCAL_TZ)

    return s.map(parse_one)


def summarize_group(g: pd.DataFrame) -> Dict[str, Any]:
    total = int(len(g))
    closed = int(g["closed"].sum())
    open_ = int((~g["closed"]).sum())
    pct_closed = (closed / total) if total else None

    ttc = g["ttc_hours"].to_numpy(dtype=float)
    age = g["open_age_hours"].to_numpy(dtype=float)

    def safe_median(arr):
        arr = arr[np.isfinite(arr)]
        return float(np.median(arr)) if arr.size else None

    def safe_mean(arr):
        arr = arr[np.isfinite(arr)]
        return float(arr.mean()) if arr.size else None

    return {
        "total": total,
        "closed": closed,
        "open": open_,
        "pct_closed": pct_closed,
        "median_ttc_hours": safe_median(ttc),
        "mean_ttc_hours": safe_mean(ttc),
        "median_open_age_hours": safe_median(age),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    feats = fetch_all_features()

    rows = []
    points: List[Optional[Point]] = []

    for ft in feats:
        props = ft.get("properties", {}) or {}
        geom = ft.get("geometry") or {}

        lon = props.get("LONGITUDE")
        lat = props.get("LATITUDE")

        if (lon is None or lat is None) and geom.get("type") == "Point":
            coords = geom.get("coordinates") or []
            if len(coords) == 2:
                lon, lat = coords[0], coords[1]

        rows.append({**props, "_lon": lon, "_lat": lat})

        if lon is None or lat is None:
            points.append(None)
        else:
            try:
                points.append(Point(float(lon), float(lat)))
            except Exception:
                points.append(None)

    df = pd.DataFrame(rows)

    df["add_dt"] = parse_dt_series(df.get("ADDDATE"))
    df["res_dt"] = parse_dt_series(df.get("RESOLUTIONDATE"))

    df["status"] = df.get("SERVICEORDERSTATUS").astype(str)
    df["closed"] = df["status"].str.lower().eq("closed")

    now = pd.Timestamp.now(tz=LOCAL_TZ)
    df["ttc_hours"] = np.where(
        df["closed"],
        (df["res_dt"] - df["add_dt"]).dt.total_seconds() / 3600.0,
        np.nan,
    )
    df["open_age_hours"] = np.where(
        ~df["closed"],
        (now - df["add_dt"]).dt.total_seconds() / 3600.0,
        np.nan,
    )
    df.loc[df["ttc_hours"] < 0, "ttc_hours"] = np.nan

    if not ANC_PATH.exists():
        raise FileNotFoundError(f"Missing {ANC_PATH}. Put your ANC GeoJSON at data/anc.geojson")
    if not WARDS_PATH.exists():
        raise FileNotFoundError(f"Missing {WARDS_PATH}. Put your Ward GeoJSON at data/wards.geojson")

    anc = load_polygons(ANC_PATH, "ANC_ID")
    wards = load_polygons(WARDS_PATH, "WARD")

    df["_lon"] = pd.to_numeric(df.get("LONGITUDE"), errors="coerce")
    df["_lat"] = pd.to_numeric(df.get("LATITUDE"), errors="coerce")

    pt_series = [
        Point(lon, lat) if (pd.notna(lon) and pd.notna(lat)) else None
        for lon, lat in zip(df["_lon"], df["_lat"])
    ]

    df["ANC_ID"] = assign_ids(pt_series, anc["polys"], anc["ids"])
    df["WARD"]   = assign_ids(pt_series, wards["polys"], wards["ids"])

    # --------
    # Write POINTS GeoJSON (minimal properties for browser filtering)
    # --------
    def to_ms(ts) -> Optional[int]:
        if ts is pd.NaT or pd.isna(ts):
            return None
        # store as UTC epoch ms for easy JS slider comparisons
        return int(ts.tz_convert("UTC").timestamp() * 1000)

    features = []
    for i, row in df.iterrows():
        lon = row.get("_lon")
        lat = row.get("_lat")
        if pd.isna(lon) or pd.isna(lat):
            continue

        add_ms = to_ms(row.get("add_dt"))
        res_ms = to_ms(row.get("res_dt"))

        # Keep properties small; avoid DETAILS (can be huge)
        props = {
        "OBJECTID": int(row["OBJECTID"]) if not pd.isna(row["OBJECTID"]) else None,
        "SERVICEREQUESTID": clean_str(row.get("SERVICEREQUESTID")),
        "SERVICECODEDESCRIPTION": clean_str(row.get("SERVICECODEDESCRIPTION")),
        "SERVICEORDERSTATUS": clean_str(row.get("SERVICEORDERSTATUS")),
        "STREETADDRESS": clean_str(row.get("STREETADDRESS")),
        "WARD": clean_str(row.get("WARD")),
        "ANC_ID": clean_str(row.get("ANC_ID")),
        "closed": clean_bool(row.get("closed")),
        "add_ms": add_ms,
        "res_ms": res_ms,
        "ttc_hours": clean_float(row.get("ttc_hours")),
        "open_age_hours": clean_float(row.get("open_age_hours")),
    }

        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
                "properties": props,
            }
        )

    points_geojson = {"type": "FeatureCollection", "features": features}
    (OUT_DIR / "points.geojson").write_text(
    json.dumps(points_geojson, ensure_ascii=False, allow_nan=False),
    encoding="utf-8",
)


    # --------
    # Aggregates (full 30 days)
    # --------
    metrics_ward = []
    for ward_key, g in df.groupby("WARD", dropna=False):
        k = None if pd.isna(ward_key) else str(ward_key)
        metrics_ward.append({"WARD": k, **summarize_group(g)})
    metrics_ward_df = pd.DataFrame(metrics_ward).sort_values("total", ascending=False)

    metrics_anc = []
    for anc_key, g in df.groupby("ANC_ID", dropna=False):
        k = None if pd.isna(anc_key) else str(anc_key)
        metrics_anc.append({"ANC_ID": k, **summarize_group(g)})
    metrics_anc_df = pd.DataFrame(metrics_anc).sort_values("total", ascending=False)

    (OUT_DIR / "metrics_ward.json").write_text(metrics_ward_df.to_json(orient="records"), encoding="utf-8")
    (OUT_DIR / "metrics_anc.json").write_text(metrics_anc_df.to_json(orient="records"), encoding="utf-8")

    ward_metric_map = {
        str(r["WARD"]): {c: (None if pd.isna(r[c]) else r[c]) for c in metrics_ward_df.columns if c != "WARD"}
        for _, r in metrics_ward_df.iterrows()
        if r["WARD"] is not None
    }
    anc_metric_map = {
        str(r["ANC_ID"]): {c: (None if pd.isna(r[c]) else r[c]) for c in metrics_anc_df.columns if c != "ANC_ID"}
        for _, r in metrics_anc_df.iterrows()
        if r["ANC_ID"] is not None
    }

    wards_choro = merge_metrics_into_geojson(wards["geojson"], "WARD", ward_metric_map)
    ancs_choro = merge_metrics_into_geojson(anc["geojson"], "ANC_ID", anc_metric_map)

    (OUT_DIR / "choropleth_wards.geojson").write_text(json.dumps(wards_choro), encoding="utf-8")
    (OUT_DIR / "choropleth_ancs.geojson").write_text(json.dumps(ancs_choro), encoding="utf-8")

    (OUT_DIR / "last_refresh.json").write_text(
        json.dumps(
            {
                "ok": True,
                "last_refresh_utc": datetime.now(timezone.utc).isoformat(),
                "layer_url": LAYER_URL,
                "records": int(len(df)),
                "generated_files": [
                    "points.geojson",
                    "metrics_ward.json",
                    "metrics_anc.json",
                    "choropleth_wards.geojson",
                    "choropleth_ancs.geojson",
                    "last_refresh.json",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Fetched {len(df)} records; wrote points + metrics to {OUT_DIR}")


if __name__ == "__main__":
    main()