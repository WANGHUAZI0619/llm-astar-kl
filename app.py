import os
from flask import Flask, render_template, request, flash
import osmnx as ox
import networkx as nx
import folium
from folium.plugins import MeasureControl, MiniMap, Fullscreen
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

# 原始经纬度图（WGS84）
G = ox.graph_from_place(PLACE, network_type="drive", simplify=True)
G = ox.distance.add_edge_lengths(G)          # 为原始图添加 length（米）

# 投影（UTM/本地米制）
Gp = ox.projection.project_graph(G)
Gp = ox.distance.add_edge_lengths(Gp)        # 为投影图再计算一次 length（米）

# -----------------------------
# 2) A* 启发函数（投影坐标，单位米）
# -----------------------------
def h(u, v):
    x1, y1 = Gp.nodes[u]["x"], Gp.nodes[u]["y"]
    x2, y2 = Gp.nodes[v]["x"], Gp.nodes[v]["y"]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

# -----------------------------
# 3) 地理编码（别名映射 + 多候选 + 国家限定）
# -----------------------------
_geolocator = Nominatim(user_agent="llm-astar-kl")
_geocode = RateLimiter(_geolocator.geocode, min_delay_seconds=1, swallow_exceptions=True)

ALIASES = {
    # 英文常见
    "klcc": "Petronas Twin Towers, Kuala Lumpur",
    "petronas": "Petronas Twin Towers, Kuala Lumpur",
    "upm": "Universiti Putra Malaysia, Serdang, Selangor",
    "mid valley": "Mid Valley Megamall, Kuala Lumpur",
    "bukit bintang": "Bukit Bintang, Kuala Lumpur",
    "sunway": "Sunway Pyramid, Subang Jaya, Selangor",
    "equine": "Equine Residences, Seri Kembangan, Selangor",
    "equine residences": "Equine Residences, Seri Kembangan, Selangor",
    # 中文别名（常用）
    "双子塔": "Petronas Twin Towers, Kuala Lumpur",
    "吉隆坡双子塔": "Petronas Twin Towers, Kuala Lumpur",
    "武吉免登": "Bukit Bintang, Kuala Lumpur",
    "双威广场": "Sunway Pyramid, Subang Jaya, Selangor",
    "布城大学": "Universiti Putra Malaysia, Serdang, Selangor",
}

def _norm(s: str) -> str:
    return (s or "").strip()

def geocode_place(q: str):
    q = _norm(q)
    if not q:
        return None

    low = q.lower()
    # 1) 别名映射
    if low in ALIASES:
        q = ALIASES[low]

    # 2) 候选组合（避免重复拼 KL/州名）
    has_my = any(k in low for k in ["malaysia", "selangor", "kuala lumpur"])
    candidates = [q]
    if not has_my:
        candidates += [f"{q}, Malaysia"]
    if "kuala lumpur" not in low:
        candidates += [f"{q}, Kuala Lumpur, Malaysia"]
    if "selangor" not in low:
        candidates += [f'{q}, Selangor, Malaysia']

    # 去重保持顺序
    seen, seq = set(), []
    for c in candidates:
        key = c.lower().strip()
        if key not in seen:
            seen.add(key)
            seq.append(c)

    # 3) 逐个尝试，限定国家 MY
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
# 5) 计算路线并生成地图（增强版可视化）
# -----------------------------
def compute_route(origin_name: str, dest_name: str):
    # 5.1 地理编码
    o = geocode_place(origin_name)
    d = geocode_place(dest_name)
    print(f"[geocode] origin={origin_name!r} -> {o}")
    print(f"[geocode] dest  ={dest_name!r} -> {d}")
    if not o or not d:
        return None, None, "Could not geocode one or both places."

    o_lat, o_lng = float(o.latitude), float(o.longitude)
    d_lat, d_lng = float(d.latitude), float(d.longitude)

    # 5.2 将经纬度点投影到 Gp 的坐标系（米）
    o_pt_proj = ox.projection.project_geometry(Point(o_lng, o_lat), to_crs=Gp.graph["crs"])[0]
    d_pt_proj = ox.projection.project_geometry(Point(d_lng, d_lat), to_crs=Gp.graph["crs"])[0]

    # 5.3 在投影图上找最近节点
    o_node = nearest_node_bruteforce(Gp, o_pt_proj.x, o_pt_proj.y)
    d_node = nearest_node_bruteforce(Gp, d_pt_proj.x, d_pt_proj.y)

    # 5.4 A* 最短路（在投影图上跑，权重 = length）
    try:
        path = nx.astar_path(Gp, o_node, d_node, heuristic=lambda u, v: h(u, v), weight="length")
        print(f"[astar] path_len={len(path)}")
    except nx.NetworkXNoPath:
        print("[astar] no path")
        return None, None, "No drivable route found."

    # 5.5 距离：逐边 length 累加（最稳妥）
    edge_lengths = ox.utils_graph.get_route_edge_attributes(Gp, path, "length")
    total_m = float(np.nansum(edge_lengths))
    if total_m <= 0 or (total_m / 1000.0) > 1000:
        # 兜底：用大圆距离按节点段相加
        tot = 0.0
        for i in range(len(path) - 1):
            n1, n2 = path[i], path[i + 1]
            lat1, lon1 = G.nodes[n1]["y"], G.nodes[n1]["x"]
            lat2, lon2 = G.nodes[n2]["y"], G.nodes[n2]["x"]
            tot += ox.distance.great_circle_vec(lat1, lon1, lat2, lon2)
        total_m = tot
    dist_km = round(total_m / 1000.0, 2)
    print(f"[dist] {dist_km} km")

    # 5.6 Folium 地图（增强款）
    # 基础底图：CartoDB + 额外 OSM/卫星可切换
    m = folium.Map(location=[(o_lat + d_lat) / 2.0, (o_lng + d_lng) / 2.0],
                   zoom_start=12, tiles="cartodb positron")
    folium.TileLayer("OpenStreetMap").add_to(m)
    folium.TileLayer("Stamen Terrain").add_to(m)

    # 起点/终点标记
    folium.Marker(
        [o_lat, o_lng],
        tooltip="Start",
        popup=folium.Popup(f"<b>Start</b><br>{origin_name}", max_width=300),
        icon=folium.Icon(color="green", icon="play"),
    ).add_to(m)

    folium.Marker(
        [d_lat, d_lng],
        tooltip="Destination",
        popup=folium.Popup(f"<b>Destination</b><br>{dest_name}", max_width=300),
        icon=folium.Icon(color="red", icon="flag"),
    ).add_to(m)

    # 路线折线 + 里程弹窗
    route_coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in path]
    folium.PolyLine(
        route_coords, weight=6, opacity=0.85, tooltip=f"{dist_km} km", color="#3465d9"
    ).add_to(m)

    # 自动聚焦到路线范围
    m.fit_bounds(route_coords)

    # 实用控件：测距 / 全屏 / 小地图
    m.add_child(MeasureControl(position="topleft", primary_length_unit="kilometers"))
    Fullscreen(position="topleft").add_to(m)
    MiniMap(toggle_display=True, minimized=True).add_to(m)

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
