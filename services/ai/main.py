"""AI Service: süre tahmini (ETA), rota optimizasyonu (OR-Tools) ve kaza sonrası yeniden rotalama.

Çalıştırma: proje kökündeki run_ai.py dosyasını çalıştır, sonra
tarayıcıda http://127.0.0.1:8000 (harita) veya /docs (API) adresini aç.
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ml.features import build_features  # noqa: E402

MODEL_DIR = ROOT / "models"
model = joblib.load(MODEL_DIR / "eta_xgb.joblib")
kmeans = joblib.load(MODEL_DIR / "kmeans_40.joblib")
FEATURES = json.loads((MODEL_DIR / "eta_features.json").read_text())

app = FastAPI(title="Rota Optimizasyonu – AI Service", version="0.2.0")
STATIC = Path(__file__).resolve().parent / "static"


# ======================= Şemalar =======================
class Point(BaseModel):
    lat: float = Field(..., ge=40.49, le=40.92, examples=[40.758])
    lon: float = Field(..., ge=-74.27, le=-73.68, examples=[-73.9855])


class Incident(BaseModel):
    lat: float
    lon: float
    radius_m: float = Field(300, ge=50, le=2000)


class EtaRequest(BaseModel):
    origin: Point
    destination: Point
    departure_time: datetime = Field(..., examples=["2016-04-12T17:30:00"])


class EtaResponse(BaseModel):
    eta_seconds: float
    eta_minutes: float


class OptimizeRequest(BaseModel):
    depot: Point
    stops: list[Point] = Field(..., min_length=1, max_length=60)
    num_vehicles: int = Field(3, ge=1, le=20)
    departure_time: datetime = Field(..., examples=["2016-04-12T17:30:00"])
    time_limit_s: int = Field(5, ge=1, le=30)
    incidents: list[Incident] = []


class VehicleState(BaseModel):
    vehicle: int
    position: Point            # aracın şu anki konumu
    remaining: list[int]       # henüz teslim edilmemiş duraklar (stops listesinde 1'den başlayan sıra)


class RerouteRequest(BaseModel):
    depot: Point
    stops: list[Point]                 # orijinal tüm duraklar
    vehicles: list[VehicleState]
    current_time: datetime             # simülasyonun şu anki saati
    incidents: list[Incident] = []
    time_limit_s: int = Field(2, ge=1, le=30)


class Leg(BaseModel):
    from_stop: int             # 0 = depo, 1..n = durak, -1 = aracın şu anki konumu
    to_stop: int
    minutes: float
    path: list[list[float]] | None = None


class VehicleRoute(BaseModel):
    vehicle: int
    stop_order: list[int]
    duration_minutes: float
    path: list[list[float]] | None = None
    legs: list[Leg] = []


class OptimizeResponse(BaseModel):
    routes: list[VehicleRoute]
    total_minutes: float
    longest_route_minutes: float
    greedy_total_minutes: float = 0
    greedy_longest_minutes: float = 0
    improvement_vs_greedy_percent: float = 0
    longest_improvement_vs_greedy_percent: float = 0
    solve_seconds: float
    street_geometry: bool = False
    blocked_road_segments: int = 0


# ======================= Yardımcılar =======================
def predict_seconds(pairs_df: pd.DataFrame) -> np.ndarray:
    feats = build_features(pairs_df, kmeans)[FEATURES]
    return np.expm1(model.predict(feats))


def eta_matrix(points: list[tuple[float, float]], when: datetime) -> np.ndarray:
    n = len(points)
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    raw = pd.DataFrame({
        "pickup_latitude": [points[i][0] for i, _ in pairs],
        "pickup_longitude": [points[i][1] for i, _ in pairs],
        "dropoff_latitude": [points[j][0] for _, j in pairs],
        "dropoff_longitude": [points[j][1] for _, j in pairs],
        "pickup_datetime": pd.to_datetime([when] * len(pairs)),
        "vendor_id": 1,
        "passenger_count": 1,
    })
    M = np.zeros((n, n), dtype=float)
    for (i, j), sec in zip(pairs, predict_seconds(raw)):
        M[i, j] = sec
    return M


def road_context(coords: list[tuple[float, float]], incidents: list[Incident]):
    """Yol ağı, en yakın kavşaklar ve kapalı yollar. Yol ağı yoksa (None, None, set())."""
    try:
        from ml.roadnet import blocked_edges, load_graph, nearest_nodes_many
        G = load_graph()
        nodes = nearest_nodes_many(G, coords)
        blocked = blocked_edges(G, [(i.lat, i.lon, i.radius_m) for i in incidents])
        return G, nodes, blocked
    except Exception as e:  # osmnx yok ya da graf indirilmedi
        print("Yol ağı kullanılamadı:", e)
        return None, None, set()


def solve_vrp(M: np.ndarray, starts: list[int], ends: list[int], time_limit_s: int,
              balance: bool = True) -> list[list[int]] | None:
    """Genel VRP: her aracın başlangıç ve bitiş düğümü ayrı verilebilir."""
    manager = pywrapcp.RoutingIndexManager(len(M), len(starts), starts, ends)
    routing = pywrapcp.RoutingModel(manager)
    Mi = np.rint(M).astype(int)

    def travel_time(a, b):
        return int(Mi[manager.IndexToNode(a)][manager.IndexToNode(b)])

    transit = routing.RegisterTransitCallback(travel_time)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)
    routing.AddDimension(transit, 0, 12 * 3600, True, "Time")
    if balance:
        routing.GetDimensionOrDie("Time").SetGlobalSpanCostCoefficient(100)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(time_limit_s)

    sol = routing.SolveWithParameters(params)
    if sol is None:
        return None
    routes = []
    for v in range(len(starts)):
        idx, r = routing.Start(v), []
        while not routing.IsEnd(idx):
            r.append(manager.IndexToNode(idx))
            idx = sol.Value(routing.NextVar(idx))
        r.append(manager.IndexToNode(idx))
        routes.append(r)
    return routes


def greedy_routes(M: np.ndarray, num_vehicles: int) -> list[list[int]]:
    unvisited = set(range(1, len(M)))
    routes = [[0] for _ in range(num_vehicles)]
    v = 0
    while unvisited:
        last = routes[v][-1]
        nxt = min(unvisited, key=lambda j: M[last][j])
        routes[v].append(nxt)
        unvisited.remove(nxt)
        v = (v + 1) % num_vehicles
    return [r + [0] for r in routes]


def route_minutes(r: list[int], M: np.ndarray) -> float:
    return sum(M[a][b] for a, b in zip(r, r[1:])) / 60


def build_routes(routes, M, coords, labels, vehicle_ids, G, nodes, blocked) -> list[VehicleRoute]:
    """Düğüm sırasından harita için bacakları ve sokak güzergâhlarını üretir.

    labels: iç düğüm indeksi -> dışarıya gösterilen durak numarası (0 depo, -1 araç konumu).
    """
    street_path = None
    if G is not None:
        from ml.roadnet import street_path
    out = []
    for vid, r in zip(vehicle_ids, routes):
        legs, full = [], []
        for i, j in zip(r, r[1:]):
            seg = street_path(G, coords[i], coords[j], nodes[i], nodes[j], blocked) if G is not None else None
            legs.append(Leg(from_stop=labels[i], to_stop=labels[j], minutes=round(M[i][j] / 60, 2), path=seg))
            if seg:
                full.extend(seg if not full else seg[1:])
        has_work = len(r) > 2 or (labels[r[0]] == -1 and len(r) > 1)
        out.append(VehicleRoute(
            vehicle=vid,
            stop_order=[labels[k] for k in r],
            duration_minutes=round(route_minutes(r, M), 1),
            path=full or None,
            legs=legs if has_work else [],
        ))
    return out


def apply_incidents(M, G, nodes, blocked):
    if G is None or not blocked:
        return M
    from ml.roadnet import detour_factors
    return M * np.array(detour_factors(G, nodes, blocked))


# ======================= Uç noktalar =======================
@app.get("/", include_in_schema=False)
def map_page():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "features": len(FEATURES)}


@app.post("/predict-eta", response_model=EtaResponse)
def predict_eta(req: EtaRequest):
    M = eta_matrix([(req.origin.lat, req.origin.lon), (req.destination.lat, req.destination.lon)], req.departure_time)
    sec = float(M[0, 1])
    return EtaResponse(eta_seconds=round(sec, 1), eta_minutes=round(sec / 60, 2))


@app.post("/optimize-routes", response_model=OptimizeResponse)
def optimize_routes(req: OptimizeRequest):
    coords = [(p.lat, p.lon) for p in [req.depot, *req.stops]]
    M = eta_matrix(coords, req.departure_time)
    G, nodes, blocked = road_context(coords, req.incidents)
    M = apply_incidents(M, G, nodes, blocked)

    t0 = time.time()
    k = req.num_vehicles
    routes = solve_vrp(M, [0] * k, [0] * k, req.time_limit_s)
    solve_s = time.time() - t0
    if routes is None:
        raise HTTPException(status_code=422, detail="Bu süre sınırı içinde geçerli bir rota bulunamadı.")

    mins = [route_minutes(r, M) for r in routes]
    g_mins = [route_minutes(r, M) for r in greedy_routes(M, k)]

    def pct(base, new):
        return round((base - new) / base * 100, 1) if base else 0.0

    labels = list(range(len(coords)))
    vr = build_routes(routes, M, coords, labels, list(range(1, k + 1)), G, nodes, blocked)
    return OptimizeResponse(
        routes=vr,
        total_minutes=round(sum(mins), 1),
        longest_route_minutes=round(max(mins), 1),
        greedy_total_minutes=round(sum(g_mins), 1),
        greedy_longest_minutes=round(max(g_mins), 1),
        improvement_vs_greedy_percent=pct(sum(g_mins), sum(mins)),
        longest_improvement_vs_greedy_percent=pct(max(g_mins), max(mins)),
        solve_seconds=round(solve_s, 2),
        street_geometry=G is not None,
        blocked_road_segments=len(blocked),
    )


@app.post("/reroute", response_model=OptimizeResponse)
def reroute(req: RerouteRequest):
    """Kaza / yeni durum sonrası: araçlar şu anki konumlarından, teslim edilmemiş durakları yeniden paylaşır."""
    remaining = sorted({s for v in req.vehicles for s in v.remaining if 1 <= s <= len(req.stops)})
    # İç düğümler: 0 = depo, sonra kalan duraklar, sonra araçların konumları
    coords = [(req.depot.lat, req.depot.lon)]
    labels = [0]
    for s in remaining:
        p = req.stops[s - 1]
        coords.append((p.lat, p.lon))
        labels.append(s)
    starts = []
    for v in req.vehicles:
        starts.append(len(coords))
        coords.append((v.position.lat, v.position.lon))
        labels.append(-1)

    M = eta_matrix(coords, req.current_time)
    G, nodes, blocked = road_context(coords, req.incidents)
    M = apply_incidents(M, G, nodes, blocked)
    # Araç başlangıç düğümlerine geri dönülmez
    for s in starts:
        M[:, s] = 10 ** 7
        M[s, s] = 0

    t0 = time.time()
    routes = solve_vrp(M, starts, [0] * len(starts), req.time_limit_s)
    solve_s = time.time() - t0
    if routes is None:
        raise HTTPException(status_code=422, detail="Yeniden rotalama başarısız oldu.")

    mins = [route_minutes(r, M) for r in routes]
    vr = build_routes(routes, M, coords, labels, [v.vehicle for v in req.vehicles], G, nodes, blocked)
    return OptimizeResponse(
        routes=vr,
        total_minutes=round(sum(mins), 1),
        longest_route_minutes=round(max(mins), 1),
        solve_seconds=round(solve_s, 2),
        street_geometry=G is not None,
        blocked_road_segments=len(blocked),
    )
