"""Manhattan yol ağı: OpenStreetMap'ten indirme ve sokak güzergâhı bulma.

Kavşaklar düğüm (node), yollar kenar (edge) olarak bir yönlü grafta tutulur.
İki nokta arasındaki güzergâh, yol uzunluğuna göre Dijkstra algoritması ile bulunur.
"""
import time
from functools import lru_cache
from pathlib import Path

import networkx as nx
import osmnx as ox

ROOT = Path(__file__).resolve().parents[1]
GRAPH_PATH = ROOT / "data" / "manhattan_drive.graphml"
PLACE = "Manhattan, New York, USA"


# Ücretsiz Overpass sunucuları sık sık yoğun olabiliyor; sırayla denenir
OVERPASS_URLS = [
    "https://overpass-api.de/api",
    "https://overpass.private.coffee/api",
    "https://maps.mail.ru/osm/tools/overpass/api",
    "https://overpass.kumi.systems/api",
]


def download_graph(path: Path = GRAPH_PATH, rounds: int = 3):
    """Yol ağını indirir ve diske kaydeder (yalnızca bir kez gerekir).

    Sunucu meşgulse (502/504/429) bekleyip tekrar dener ve yedek sunuculara geçer.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    wait = 10
    last_error = None
    G = None
    for attempt in range(1, rounds + 1):
        for url in OVERPASS_URLS:
            ox.settings.overpass_url = url
            try:
                print(f"  Deneme {attempt}: {url}")
                G = ox.graph_from_place(PLACE, network_type="drive")
                break
            except Exception as e:  # sunucu meşgul / zaman aşımı
                last_error = e
                print(f"    olmadı: {str(e).splitlines()[0][:100]}")
        if G is not None:
            break
        print(f"  Tüm sunucular meşgul, {wait} sn bekleniyor...")
        time.sleep(wait)
        wait *= 2   # üstel geri çekilme (exponential backoff)
    if G is None:
        raise RuntimeError("Yol ağı indirilemedi; biraz sonra tekrar dene.") from last_error
    # Tek yönlü sokaklarda her yerden her yere gidilebilsin diye
    # en büyük güçlü bağlı bileşeni tut
    G = ox.truncate.largest_component(G, strongly=True)
    ox.save_graphml(G, path)
    return G


@lru_cache(maxsize=1)
def load_graph():
    if not GRAPH_PATH.exists():
        raise FileNotFoundError(
            f"Yol ağı bulunamadı: {GRAPH_PATH}. Önce 'python prepare_roadnet.py' çalıştır."
        )
    return ox.load_graphml(GRAPH_PATH)


def nearest_node(G, lat: float, lon: float) -> int:
    return ox.distance.nearest_nodes(G, X=lon, Y=lat)


# ---------- Kaza / kapalı yol desteği ----------
BLOCK_PENALTY = 1000      # kapalı yolun Dijkstra'daki bedel çarpanı
UNAVOIDABLE_DELAY = 3.0   # kazadan kaçınılamıyorsa süre çarpanı


def _dist_m(lat1, lon1, lat2, lon2):
    import math
    p = math.pi / 180
    a = (math.sin((lat2 - lat1) * p / 2) ** 2
         + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lon2 - lon1) * p / 2) ** 2)
    return 12742000 * math.asin(math.sqrt(a))


def blocked_edges(G, incidents: list[tuple[float, float, float]]) -> set[tuple[int, int]]:
    """Kaza alanı içinde kalan yol parçaları. incidents: [(lat, lon, yarıçap_m), ...]"""
    blocked: set[tuple[int, int]] = set()
    if not incidents:
        return blocked
    for u, v in G.edges():
        y = (G.nodes[u]["y"] + G.nodes[v]["y"]) / 2
        x = (G.nodes[u]["x"] + G.nodes[v]["x"]) / 2
        if any(_dist_m(y, x, lat, lon) <= r for lat, lon, r in incidents):
            blocked.add((u, v))
    return blocked


def _edge_len(edge_dict) -> float:
    return min(d.get("length", 1.0) for d in edge_dict.values())


def make_weight(blocked: set):
    def w(u, v, edge_dict):
        base = _edge_len(edge_dict)
        return base * BLOCK_PENALTY if (u, v) in blocked else base
    return w


def detour_factors(G, nodes: list[int], blocked: set) -> list[list[float]]:
    """Her nokta çifti için 'kazalı güzergâh süresi / normal süre' çarpanı.

    Kaza yoksa hepsi 1'dir. Kazadan kaçmak için yapılan dolambaç oranında süre artar;
    kaza alanından geçmek zorunluysa ek gecikme uygulanır.
    """
    n = len(nodes)
    F = [[1.0] * n for _ in range(n)]
    if not blocked:
        return F
    w = make_weight(blocked)
    for i, s in enumerate(nodes):
        free = nx.single_source_dijkstra_path_length(G, s, weight="length")
        _, paths = nx.single_source_dijkstra(G, s, weight=w)
        for j, t in enumerate(nodes):
            if i == j or t not in paths or free.get(t, 0) <= 0:
                continue
            p = paths[t]
            real = sum(_edge_len(G.get_edge_data(a, b)) for a, b in zip(p, p[1:]))
            factor = real / free[t]
            if any((a, b) in blocked for a, b in zip(p, p[1:])):
                factor *= UNAVOIDABLE_DELAY
            F[i][j] = max(1.0, factor)
    return F


def nearest_nodes_many(G, coords: list[tuple[float, float]]) -> list[int]:
    """Tüm noktaların en yakın kavşaklarını tek seferde bulur (çok daha hızlı)."""
    return list(ox.distance.nearest_nodes(G, X=[c[1] for c in coords], Y=[c[0] for c in coords]))


def street_path(G, a: tuple[float, float], b: tuple[float, float],
                na: int | None = None, nb: int | None = None,
                blocked: set | None = None) -> list[list[float]]:
    """a ve b (lat, lon) arasındaki sokak güzergâhını [[lat, lon], ...] olarak döndürür.

    na / nb verilirse en yakın kavşak araması atlanır. blocked verilirse kapalı yollardan kaçınır.
    """
    na = nearest_node(G, *a) if na is None else na
    nb = nearest_node(G, *b) if nb is None else nb
    weight = make_weight(blocked) if blocked else "length"
    try:
        nodes = nx.shortest_path(G, na, nb, weight=weight)
    except nx.NetworkXNoPath:
        return [list(a), list(b)]
    coords = [list(a)]
    for u, v in zip(nodes, nodes[1:]):
        data = min(G.get_edge_data(u, v).values(), key=lambda d: d.get("length", 0))
        if "geometry" in data:                       # kıvrımlı yollar
            coords.extend([[y, x] for x, y in data["geometry"].coords])
        else:
            coords.append([G.nodes[u]["y"], G.nodes[u]["x"]])
            coords.append([G.nodes[v]["y"], G.nodes[v]["x"]])
    coords.append(list(b))
    return coords


def route_geometry(G, points: list[tuple[float, float]], order: list[int]) -> list[list[float]]:
    """Durak sırasına göre (0 = depo) tüm rotanın sokak güzergâhını birleştirir."""
    path: list[list[float]] = []
    for i, j in zip(order, order[1:]):
        seg = street_path(G, points[i], points[j])
        path.extend(seg if not path else seg[1:])
    return path
