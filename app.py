import os
from flask import Flask, render_template, request, flash
import osmnx as ox
import networkx as nx
import folium
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from shapely.geometry import Point
import numpy as np

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-key")

# -----------------------------
# 1) 构建图：原始经纬度图 G（画地图用）+ 投影图 Gp（路径搜索与距离计算用，单位米）
# -----------------------------
PLACE = "Kuala Lumpur, Malaysia"
print("Building road graph for:", PLACE)

# 原始经纬度图
G = ox.graph_from_place(PLACE, network_type="drive", simplify=True)
G = ox.distance.add_edge_lengths(G)          # 给边添加 length（米）

# 投影到米制坐标的图
Gp = ox.projection.project_graph(G)
Gp = ox.distance.add_edge_lengths(Gp)        # 保底再算一次 length（米）

# -----------------------------
# 2) A* 启发函数（投影坐标，单位米）
# -----------------------------
def h(u, v):
    x1, y1 = Gp.nodes[u]["x"], Gp.nodes[u]["y"]
    x2, y2 = Gp.nodes[v]["x"], Gp.nodes[v]["y"]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

# -----------------------------
# 3) 地理编码（带限速）
# -----------------------------
_geolocator = Nominatim(user_agent="llm-astar-kl")
_geocode = RateLimiter(_geolocator.geocode, min_delay_seconds=1, swallow_exceptions=False)

def _norm(q: str) -> str:
    return (q or "").strip()

# 改进版：别名映射 + 多候选 + 限定国家为马来西亚（MY）
def geocode_place(q: str):
    q = _norm(q)
    if not q:
        return None

    low = q.lower()

    # 常见别名（可按需扩展）
    alias = {
        "klcc": "Petronas Twin Towers, Kuala Lumpur",
        "upm": "Universiti Putra Malaysia, Serdang, Selangor",
        "mid valley": "Mid Valley Megamall, Kuala Lumpur",
        "bukit bintang": "Bukit Bintang, Kuala Lumpur",
        "sunway": "Sunway Pyramid, Subang Jaya, Selangor",
        "equine": "Equine Residences, Seri Kembangan, Selangor",
        "equine residences": "Equine Residences, Seri Kembangan, Selangor",
        "petronas": "Petronas Twin Towers, Kuala Lumpur",
    }
    if low in alias:
        q = alias[low]

    # 是否已包含国家/州信息
    has_my = any(k in low for k in ["malaysia", "selangor", "kuala lumpur"])

    # 依次尝试的候选
    candidates = []
    candidates.append(q)  # 原样
    if not has_my:
        candidates.append(f"{q}, Malaysia")
    if "kuala lumpur" not in low:
        candidates.append(f"{q}, Kuala Lumpur, Malaysia")
    if "selangor" not in low:
        candidates.append(f"{q}, Selangor, Malaysia")

    # 去重保持顺序
    seen = set()
    seq = []
    for c in candidates:
        key = c.lower().strip()
        if key not in seen:
            seen.add(key)
            seq.append(c)

    # 逐个尝试；限定只在马来西亚搜
    for cand in seq:
        loc = _geocode(cand, country_codes="my", exactly_one=True)
        if loc:
            return loc

    return None

# -----------------------------
# 4) 最近节点：NumPy 暴力最近点（免 scipy/sklearn）
# -----------------------------
_nodes_cache = None  # (nodes, xs, ys)

def nearest_node_bruteforce(graph_proj, x_m, y_m):
    """在投影图 graph_proj（米制）上，从 (x_m, y_m) 找最近节点ID。"""
    global _nodes_cache
    if _nodes_cache is None:
        nodes = np.fromiter(graph_proj.nodes(), dtype=object)
        xs = np.array([graph_proj.nodes[n]["x"] for n in nodes], dtype=float)
        ys = np.array([graph_proj.nodes[n]["y"] for n in nodes], dtype=float)
        _nodes_cache = (nodes, xs, ys)
    else:
        nodes, xs, ys = _nodes_cache
    idx = np.argmin((xs - x_m) ** 2 + (ys - y_m) ** 2)
    return nodes[idx]

# -----------------------------
# 5) 计算路线并生成地图
# -----------------------------
def compute_route(origin_name: str, dest_name: str):
    # 5.1 地理编码
    o = geocode_place(origin_name)
    d = geocode_place(dest_name)
    if not o or not d:
        return None, None, "Could not geocode one or both places."

    o_lat, o_lng = float(o.latitude), float(o.longitude)
    d_lat, d_lng = float(d.latitude), float(d.longitude)

    # 5.2 将经纬度点投影到 Gp 的坐标系（米）
    o_pt_proj = ox.projection.project_geometry(Point(o_lng, o_lat), to_crs=Gp.graph["crs"])[0]
    d_pt_proj = ox.projection.project_geometry(Point(d_lng, d_lat), to_crs=Gp.graph["crs"])[0]

    # 5.3 在投影图上找最近节点（NumPy 暴力法）
    o_node = nearest_node_bruteforce(Gp, o_pt_proj.x, o_pt_proj.y)
    d_node = nearest_node_bruteforce(Gp, d_pt_proj.x, d_pt_proj.y)

    # 5.4 A* 最短路（在投影图上跑，权重 = length）
    try:
        path = nx.astar_path(Gp, o_node, d_node, heuristic=lambda u, v: h(u, v), weight="length")
    except nx.NetworkXNoPath:
        return None, None, "No drivable route found."

    # 5.5 距离：逐边 length 累加（最稳妥）
    edge_lengths = ox.utils_graph.get_route_edge_attributes(Gp, path, "length")
    total_m = float(np.nansum(edge_lengths))

    # 兜底：若异常（<=0 或 >1000km），用大圆距离按节点段落相加
    if total_m <= 0 or (total_m / 1000.0) > 1000:
        tot = 0.0
        for i in range(len(path) - 1):
            n1, n2 = path[i], path[i + 1]
            lat1, lon1 = G.nodes[n1]["y"], G.nodes[n1]["x"]
            lat2, lon2 = G.nodes[n2]["y"], G.nodes[n2]["x"]
            tot += ox.distance.great_circle_vec(lat1, lon1, lat2, lon2)
        total_m = tot

    dist_km = round(total_m / 1000.0, 2)

    # 5.6 Folium 地图（画线用原始经纬度图 G 的坐标）
    m = folium.Map(location=[o_lat, o_lng], zoom_start=12)
    folium.Marker([o_lat, o_lng], tooltip="Origin", popup=origin_name).add_to(m)
    folium.Marker([d_lat, d_lng], tooltip="Destination", popup=dest_name).add_to(m)
    nodes_latlng = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in path]
    folium.PolyLine(nodes_latlng, weight=6, opacity=0.8).add_to(m)
    folium.FitBounds(nodes_latlng).add_to(m)

    return dist_km, m._repr_html_(), None

# -----------------------------
# 6) Flask 路由
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    map_html = None
    distance_km = None
    origin = request.form.get("origin", "")
    dest = request.form.get("dest", "")

    if request.method == "POST":
        if not origin or not dest:
            flash("Please fill in both Point A and Point B.")
        else:
            distance_km, map_html, err = compute_route(origin.strip(), dest.strip())
            if err:
                flash(err)

    return render_template(
        "index.html",
        map_html=map_html,
        distance_km=distance_km,
        origin_val=origin,
        dest_val=dest,
    )

if __name__ == "__main__":
    app.run(debug=True)
