import sys
import math
import folium
import xml.etree.ElementTree as ET
import heapq
import json

###############################################################################
# 1. Parse assignment file (6-column WAYS: time in minutes, directed edges)
###############################################################################

def split_csv_allow_commas(line, min_fields):
    parts = []
    buf = []
    depth = 0
    for ch in line:
        if ch == '(':
            depth += 1
            buf.append(ch)
        elif ch == ')':
            depth = max(depth - 1, 0)
            buf.append(ch)
        elif ch == ',':
            if depth == 0:
                parts.append("".join(buf).strip())
                buf = []
            else:
                buf.append(ch)
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf).strip())
    if len(parts) < min_fields:
        raise ValueError(f"Line '{line}' parsed into too few fields: {parts}")
    return parts


def parse_assignment_file(path):
    section = None
    nodes = {}
    ways = []
    cameras = {}
    meta = {"start": None, "goals": [], "accident_multiplier": None}

    def is_header(line):
        return line.startswith("[") and line.endswith("]")

    def ignore(line):
        return (not line.strip()) or line.strip().startswith("#")

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if ignore(line):
                continue
            if is_header(line):
                section = line.upper()
                continue

            if section == "[NODES]":
                p = split_csv_allow_commas(line, 4)
                nid, lat, lon, label = p[0], float(p[1]), float(p[2]), p[3]
                nodes[nid] = {"lat": lat, "lon": lon, "label": label}

            elif section == "[WAYS]":
                # way_id,from_node,to_node,road_name,highway_type,travel_time_min
                p = split_csv_allow_commas(line, 6)
                ways.append({
                    "way_id": p[0],
                    "from": p[1],
                    "to": p[2],
                    "road_name": p[3],
                    "highway_type": p[4],
                    "time_min": float(p[5]),
                })

            elif section == "[CAMERAS]":
                p = split_csv_allow_commas(line, 2)
                cameras[p[0]] = p[1]

            elif section == "[META]":
                p = [x.strip() for x in line.split(",")]
                key = p[0].upper()
                if key == "START":
                    meta["start"] = p[1]
                elif key == "GOAL":
                    meta["goals"] = p[1:]
                elif key == "ACCIDENT_MULTIPLIER":
                    meta["accident_multiplier"] = float(p[1])

    return nodes, ways, cameras, meta

###############################################################################
# 2. Load map.osm into a road graph
###############################################################################

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    la1, la2 = math.radians(lat1), math.radians(lat2)
    dla = la2 - la1
    dlo = math.radians(lon2 - lon1)
    a = math.sin(dla/2)**2 + math.cos(la1)*math.cos(la2)*math.sin(dlo/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def load_osm_graph(osm_path):
    tree = ET.parse(osm_path)
    root = tree.getroot()

    osm_nodes = {}
    for n in root.findall("node"):
        nid = n.attrib["id"]
        lat = float(n.attrib["lat"])
        lon = float(n.attrib["lon"])
        osm_nodes[nid] = (lat, lon)

    graph = {nid: [] for nid in osm_nodes}

    for w in root.findall("way"):
        nd_refs = [nd.attrib["ref"] for nd in w.findall("nd")]
        tags = {t.attrib.get("k"): t.attrib.get("v") for t in w.findall("tag")}
        if "highway" not in tags:
            continue

        for i in range(len(nd_refs) - 1):
            a, b = nd_refs[i], nd_refs[i+1]
            if a in osm_nodes and b in osm_nodes:
                lat1, lon1 = osm_nodes[a]
                lat2, lon2 = osm_nodes[b]
                dist = haversine_km(lat1, lon1, lat2, lon2)
                graph[a].append((b, dist))
                graph[b].append((a, dist))

    # Remove isolated nodes
    isolated = [nid for nid, nbrs in graph.items() if len(nbrs) == 0]
    for nid in isolated:
        del graph[nid]

    return osm_nodes, graph

###############################################################################
# 3. K-nearest snapping
###############################################################################

def k_nearest_graph_nodes(lat, lon, osm_nodes, graph, k=5):
    candidates = []
    for nid, (nlat, nlon) in osm_nodes.items():
        if nid not in graph:
            continue
        d_lat = lat - nlat
        d_lon = lon - nlon
        d2 = d_lat*d_lat + d_lon*d_lon
        candidates.append((d2, nid))
    candidates.sort(key=lambda x: x[0])
    return [nid for d2, nid in candidates[:k]]

###############################################################################
# 4. Dijkstra in road graph
###############################################################################

def dijkstra_path(graph, start_id, goal_id):
    dist = {start_id: 0.0}
    prev = {}
    pq = [(0.0, start_id)]
    visited = set()

    while pq:
        cur_d, cur = heapq.heappop(pq)
        if cur in visited:
            continue
        visited.add(cur)
        if cur == goal_id:
            break

        for nbr, w in graph.get(cur, []):
            nd = cur_d + w
            if nbr not in dist or nd < dist[nbr]:
                dist[nbr] = nd
                prev[nbr] = cur
                heapq.heappush(pq, (nd, nbr))

    if goal_id not in dist:
        return None

    path = []
    node = goal_id
    while True:
        path.append(node)
        if node == start_id:
            break
        node = prev.get(node)
        if node is None:
            return None
    path.reverse()
    return path


def path_length_km(path, graph):
    if not path or len(path) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(path[:-1], path[1:]):
        for nbr, w in graph.get(a, []):
            if nbr == b:
                total += w
                break
    return total

###############################################################################
# 5. Folium helpers
###############################################################################

def color_for_highway(hwy_type):
    h = (hwy_type or "").lower()
    if h == "primary":
        return "deepskyblue"
    elif h == "secondary":
        return "purple"
    elif h == "tertiary":
        return "darkblue"
    elif h == "service":
        return "slategray"
    else:
        return "orange"  # fallback color (was white, invisible)


def get_map_center(nodes):
    lats = [info["lat"] for info in nodes.values()]
    lons = [info["lon"] for info in nodes.values()]
    return sum(lats)/len(lats), sum(lons)/len(lons)

###############################################################################
# 6. Main visualiser
###############################################################################

def visualize_with_roads(nodes, ways, cameras, meta,
                         osm_nodes, graph,
                         out_html="heritage_map_roads.html",
                         k_snap=5):

    center_lat, center_lon = get_map_center(nodes)
    m = folium.Map(location=[center_lat, center_lon],
                   zoom_start=18,
                   tiles="OpenStreetMap")

    # Precompute snap candidates
    snap_candidates = {}
    for nid, info in nodes.items():
        snap_candidates[nid] = k_nearest_graph_nodes(
            info["lat"], info["lon"], osm_nodes, graph, k=k_snap
        )

    print("[DEBUG] Snap candidates (first 3 per node):")
    for nid, cand in snap_candidates.items():
        print(f"  Node {nid} ({nodes[nid]['label']}): {cand[:3]}")

    print(f"\n[DEBUG] Drawing {len(ways)} ways...")
    # Draw each edge following road geometry if possible
    for w in ways:
        way_id = w["way_id"]
        u = w["from"]
        v = w["to"]
        rn = w["road_name"]
        hwy = w["highway_type"]
        time_min = w["time_min"]

        best_path = None
        best_len = None

        # Check if snap candidates overlap (nodes very close together)
        u_snaps = set(snap_candidates[u])
        v_snaps = set(snap_candidates[v])
        is_straight_line = False
        if u_snaps & v_snaps:  # If any overlap, use straight line
            latlngs = [(nodes[u]["lat"], nodes[u]["lon"]),
                       (nodes[v]["lat"], nodes[v]["lon"])]
            is_straight_line = True
            print(f"  Way {way_id} ({u}→{v}): OVERLAP - using straight line, {len(latlngs)} points")
        else:
            for u_osm in snap_candidates[u]:
                for v_osm in snap_candidates[v]:
                    path = dijkstra_path(graph, u_osm, v_osm)
                    if path is None:
                        continue
                    plen = path_length_km(path, graph)
                    if best_len is None or plen < best_len:
                        best_len = plen
                        best_path = path

            if best_path is None or len(best_path) < 2:
                latlngs = [(nodes[u]["lat"], nodes[u]["lon"]),
                           (nodes[v]["lat"], nodes[v]["lon"])]
                is_straight_line = True
                print(f"  Way {way_id} ({u}→{v}): NO PATH - using straight line, {len(latlngs)} points")
            else:
                latlngs = [(osm_nodes[nid][0], osm_nodes[nid][1]) for nid in best_path]
                print(f"  Way {way_id} ({u}→{v}): ROUTED via {len(best_path)} OSM nodes, {len(latlngs)} points")

        is_camera = way_id in cameras
        if is_camera:
            line_color = "crimson"
            weight = 6
            dash_array = "8,4"
            opacity = 0.9
        elif is_straight_line:
            # Straight line fallback (overlapping nodes or no OSM path)
            line_color = "gray"
            weight = 3
            dash_array = "4,8"
            opacity = 0.7
        else:
            line_color = color_for_highway(hwy)
            weight = 5 if hwy in ["primary", "secondary"] else 4
            dash_array = None
            opacity = 0.8

        tooltip_text = f"{rn} ({time_min} min)"
        popup_html = (
            f"<b>{rn}</b><br>"
            f"way_id: {way_id}<br>"
            f"type: {hwy}<br>"
            f"time: {time_min} min<br>"
        )
        if is_camera:
            popup_html += "<b>CAMERA MONITORED</b><br>Accident → time multiplied"

        polyline = folium.PolyLine(
            locations=latlngs,
            color=line_color,
            weight=weight,
            opacity=opacity,
            dash_array=dash_array,
            tooltip=tooltip_text,
            popup=folium.Popup(popup_html, max_width=250),
            class_name="assignment-edge"
        )
        polyline.add_to(m)

    # Node markers
    start_node = meta["start"]
    goal_nodes = set(meta["goals"])

    for nid, info in nodes.items():
        lat, lon, label = info["lat"], info["lon"], info["label"]

        if nid == start_node:
            fill_color = "green"
            role = "START"
        elif nid in goal_nodes:
            fill_color = "blue"
            role = "GOAL"
        else:
            fill_color = "white"
            role = "Node"

        folium.CircleMarker(
            location=(lat, lon),
            radius=8,
            color="black",
            weight=1,
            fill=True,
            fill_color=fill_color,
            fill_opacity=0.9,
            popup=folium.Popup(
                f"<b>Node {nid}: {label}</b><br>"
                f"lat: {lat:.6f}<br>lon: {lon:.6f}<br>{role}",
                max_width=250
            ),
            tooltip=f"{nid}: {label} ({role})"
        ).add_to(m)

        folium.map.Marker(
            [lat, lon],
            icon=folium.DivIcon(
                html=f"""
                <div style="
                    font-size:10px;
                    font-weight:bold;
                    color:black;
                    background-color:rgba(255,255,255,0.8);
                    border:1px solid black;
                    border-radius:3px;
                    padding:2px 3px;
                    white-space:nowrap;
                ">{nid}: {label}</div>
                """
            )
        ).add_to(m)

    # Legend
    acc_mult = meta["accident_multiplier"]
    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 20px;
        left: 20px;
        z-index: 9999;
        background: rgba(255,255,255,0.9);
        padding: 8px 12px;
        border: 1px solid #333;
        border-radius: 4px;
        font-size: 12px;
        line-height: 1.4;
    ">
    <b>Kuching Heritage Graph</b><br>
    <span style="color:green;font-weight:bold;">●</span> START ({start_node})<br>
    <span style="color:blue;font-weight:bold;">●</span> GOAL(s) {', '.join(goal_nodes)}<br>
    <span style="color:crimson;font-weight:bold;">▬ ▬</span> Camera road (accident → time ×{acc_mult})<br>
    <span style="color:deepskyblue;font-weight:bold;">▬</span> Primary road<br>
    <span style="color:purple;font-weight:bold;">▬</span> Secondary road<br>
    <span style="color:darkblue;font-weight:bold;">▬</span> Tertiary road<br>
    <span style="color:slategray;font-weight:bold;">▬</span> Service/local<br>
    <span style="color:gray;font-weight:bold;">- -</span> Straight line (nodes too close)<br>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # -------------------------------------------------------------
    # Embed data + add collapsible client-side route planner (JS)
    # -------------------------------------------------------------
    # Prepare compact JSON payloads for client-side routing
    map_name = m.get_name()

    assign_nodes_js = {nid: {"lat": info["lat"], "lon": info["lon"], "label": info["label"]}
                       for nid, info in nodes.items()}
    snap_js = {nid: cand for nid, cand in snap_candidates.items()}
    osm_nodes_js = {nid: [latlon[0], latlon[1]] for nid, latlon in osm_nodes.items() if nid in graph}
    graph_js = {nid: [[nbr, float(w)] for (nbr, w) in nbrs] for nid, nbrs in graph.items()}
    ways_js = {w["way_id"]: {"from": w["from"], "to": w["to"], "time_min": w["time_min"], 
                             "road_name": w["road_name"]} for w in ways}

    control_html = f"""
        <style>
            .route-control {{
                position: fixed;
                top: 20px;
                right: 20px;
                z-index: 10000;
                width: 320px;
                font-family: Arial, sans-serif;
            }}
            .route-control details {{
                background: rgba(255,255,255,0.95);
                border: 1px solid #333;
                border-radius: 6px;
                padding: 8px 10px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.15);
            }}
            .route-control summary {{
                cursor: pointer;
                font-weight: bold;
            }}
            .route-control textarea {{
                width: 100%;
                min-height: 58px;
                resize: vertical;
                font-family: monospace;
            }}
            .route-control .buttons {{ display: flex; gap: 8px; margin-top: 8px; }}
            .route-control button {{ cursor: pointer; padding: 6px 10px; }}
            .route-control .hint {{ color: #444; font-size: 12px; margin-top: 6px; }}
            .route-control .error {{ color: #b00020; font-size: 12px; margin-top: 6px; }}
            .route-control .ok {{ color: #006400; font-size: 12px; margin-top: 6px; }}
        </style>
        <div class="route-control">
            <details open>
                <summary>Route planner (enter nodes)</summary>
                <div style="margin-top:8px;">
                    <label for="routeNodesInput"><b>Node order</b> (comma-separated):</label>
                    <textarea id="routeNodesInput" placeholder="e.g. 1,5,7,10"></textarea>
                    <div class="hint">Use assignment node IDs (numbers 1-15), in travel order from origin to destination.</div>
                    <div class="buttons">
                        <button id="plotRouteBtn">Plot route</button>
                        <button id="clearRouteBtn" type="button">Clear</button>
                    </div>
                    <div id="routeStatus" class="hint"></div>
                </div>
            </details>
        </div>
        <script>
            // Data from Python
            const ASSIGN_NODES = {json.dumps(assign_nodes_js)};
            const SNAP_CANDIDATES = {json.dumps(snap_js)};
            const OSM_NODES = {json.dumps(osm_nodes_js)};
            const GRAPH = {json.dumps(graph_js)};
            const WAYS = {json.dumps(ways_js)};

            // Get Folium Leaflet map object
            let MAP = null;
            function getMap() {{
                if (!MAP) {{
                    const mapName = {json.dumps(map_name)};
                    console.log('Looking for map with name:', mapName);
                    MAP = window[mapName];
                    if (!MAP) {{
                        console.error('Map not found! Available window properties:', Object.keys(window).filter(k => k.includes('map')));
                    }} else {{
                        console.log('Map found:', MAP);
                    }}
                }}
                return MAP;
            }}

            // Simple binary heap for Dijkstra
            class MinHeap {{
                constructor() {{ this._h = []; }}
                _swap(i, j) {{ const t = this._h[i]; this._h[i] = this._h[j]; this._h[j] = t; }}
                _up(i) {{
                    while (i > 0) {{
                        const p = (i - 1) >> 1;
                        if (this._h[p][0] <= this._h[i][0]) break;
                        this._swap(i, p); i = p;
                    }}
                }}
                _down(i) {{
                    const n = this._h.length;
                    while (true) {{
                        let l = i*2+1, r = l+1, m = i;
                        if (l < n && this._h[l][0] < this._h[m][0]) m = l;
                        if (r < n && this._h[r][0] < this._h[m][0]) m = r;
                        if (m === i) break;
                        this._swap(i, m); i = m;
                    }}
                }}
                push(x) {{ this._h.push(x); this._up(this._h.length-1); }}
                pop() {{
                    if (this._h.length === 0) return null;
                    const top = this._h[0];
                    const last = this._h.pop();
                    if (this._h.length) {{ this._h[0] = last; this._down(0); }}
                    return top;
                }}
                get length() {{ return this._h.length; }}
            }}

            function dijkstraPath(startId, goalId) {{
                if (!(startId in GRAPH) || !(goalId in GRAPH)) return null;
                const dist = new Map([[startId, 0]]);
                const prev = new Map();
                const visited = new Set();
                const pq = new MinHeap();
                pq.push([0, startId]);
                while (pq.length) {{
                    const [d, u] = pq.pop();
                    if (visited.has(u)) continue;
                    visited.add(u);
                    if (u === goalId) break;
                    const nbrs = GRAPH[u] || [];
                    for (let i=0;i<nbrs.length;i++) {{
                        const v = nbrs[i][0];
                        const w = nbrs[i][1];
                        const nd = d + w;
                        if (!dist.has(v) || nd < dist.get(v)) {{
                            dist.set(v, nd);
                            prev.set(v, u);
                            pq.push([nd, v]);
                        }}
                    }}
                }}
                if (!dist.has(goalId)) return null;
                const path = [];
                let cur = goalId;
                while (true) {{
                    path.push(cur);
                    if (cur === startId) break;
                    cur = prev.get(cur);
                    if (cur === undefined) return null;
                }}
                path.reverse();
                return path;
            }}

            function pathLengthKm(path) {{
                if (!path || path.length < 2) return 0;
                let total = 0;
                for (let i=0;i<path.length-1;i++) {{
                    const u = path[i], v = path[i+1];
                    const nbrs = GRAPH[u] || [];
                    for (let j=0;j<nbrs.length;j++) {{
                        if (nbrs[j][0] === v) {{ total += nbrs[j][1]; break; }}
                    }}
                }}
                return total;
            }}

            const pairCache = new Map(); // key: "U|V" (assignment IDs) -> array of [lat,lon]
            function bestLatLngsBetween(assignU, assignV) {{
                const key = assignU+"|"+assignV;
                if (pairCache.has(key)) return pairCache.get(key);
                const cu = SNAP_CANDIDATES[assignU] || [];
                const cv = SNAP_CANDIDATES[assignV] || [];
                
                // Check for overlapping snap candidates (nodes very close)
                const cuSet = new Set(cu);
                const overlap = cv.some(v => cuSet.has(v));
                
                let latlngs;
                if (overlap) {{
                    console.log(`Nodes ${{assignU}} → ${{assignV}} have overlapping snaps, using straight line`);
                    latlngs = [ [ASSIGN_NODES[assignU].lat, ASSIGN_NODES[assignU].lon],
                               [ASSIGN_NODES[assignV].lat, ASSIGN_NODES[assignV].lon] ];
                }} else {{
                    let bestPath = null, bestLen = Infinity;
                    for (let i=0;i<cu.length;i++) {{
                        for (let j=0;j<cv.length;j++) {{
                            const p = dijkstraPath(cu[i], cv[j]);
                            if (!p || p.length < 2) continue;
                            const L = pathLengthKm(p);
                            if (L < bestLen) {{ bestLen = L; bestPath = p; }}
                        }}
                    }}
                    
                    if (!bestPath || bestPath.length < 2) {{
                        console.log(`No valid OSM path for ${{assignU}} → ${{assignV}}, using straight line`);
                        latlngs = [ [ASSIGN_NODES[assignU].lat, ASSIGN_NODES[assignU].lon],
                                   [ASSIGN_NODES[assignV].lat, ASSIGN_NODES[assignV].lon] ];
                    }} else {{
                        latlngs = bestPath.map(nid => [OSM_NODES[nid][0], OSM_NODES[nid][1]]);
                        console.log(`Routed ${{assignU}} → ${{assignV}} via ${{bestPath.length}} OSM nodes`);
                    }}
                }}
                pairCache.set(key, latlngs);
                return latlngs;
            }}

            let currentRoute = null;
            function clearRoute() {{
                const map = getMap();
                if (currentRoute && map) {{ 
                    map.removeLayer(currentRoute); 
                    currentRoute = null; 
                }}
                // Restore all assignment edges
                document.querySelectorAll('.assignment-edge').forEach(el => {{
                    el.style.display = '';
                }});
            }}

            function plotRouteFromInput() {{
                const status = document.getElementById('routeStatus');
                status.className = 'hint';
                status.textContent = '';
                const raw = (document.getElementById('routeNodesInput').value || '').trim();
                if (!raw) {{ status.textContent = 'Enter node IDs, e.g. 1,5,7'; return; }}
                // Strip optional 'N' prefix if present (e.g., "N1" -> "1")
                const ids = raw.split(',').map(s=>s.trim().replace(/^[Nn]/, '')).filter(Boolean);
                console.log('Plotting route for nodes:', ids);
                const missing = ids.filter(id => !(id in ASSIGN_NODES));
                if (missing.length) {{
                    status.className = 'error';
                    status.textContent = 'Unknown node ID(s): ' + missing.join(', ');
                    console.error('Unknown node IDs:', missing);
                    return;
                }}
                
                // Calculate total travel time
                let totalTime = 0;
                for (let i=0;i<ids.length-1;i++) {{
                    const u = ids[i];
                    const v = ids[i+1];
                    // Find matching way
                    let found = false;
                    for (const wayId in WAYS) {{
                        const way = WAYS[wayId];
                        if (way.from === u && way.to === v) {{
                            totalTime += way.time_min;
                            found = true;
                            console.log(`Segment ${{u}} → ${{v}}: ${{way.time_min}} min (${{way.road_name}})`);
                            break;
                        }}
                    }}
                    if (!found) {{
                        console.warn(`No way found for ${{u}} → ${{v}}`);
                    }}
                }}
                
                let acc = [];
                for (let i=0;i<ids.length-1;i++) {{
                    const seg = bestLatLngsBetween(ids[i], ids[i+1]);
                    console.log(`Segment ${{ids[i]}} → ${{ids[i+1]}}: ${{seg.length}} points`, seg.slice(0,2));
                    if (acc.length && seg.length) {{ acc = acc.concat(seg.slice(1)); }}
                    else {{ acc = acc.concat(seg); }}
                }}
                console.log('Total route points:', acc.length);
                console.log('First 3 points:', acc.slice(0,3));
                console.log('Last 3 points:', acc.slice(-3));
                if (acc.length < 2) {{ 
                    status.className = 'error';
                    status.textContent = 'Not enough points to draw route.';
                    console.error('Insufficient points:', acc);
                    return;
                }}
                clearRoute();
                const map = getMap();
                if (!map) {{
                    status.className = 'error';
                    status.textContent = 'Map not initialized. Please reload the page.';
                    console.error('MAP is null!');
                    return;
                }}
                
                // Hide all assignment edges
                document.querySelectorAll('.assignment-edge').forEach(el => {{
                    el.style.display = 'none';
                }});
                
                console.log('Creating polyline with', acc.length, 'points...');
                console.log('MAP object:', map);
                console.log('L (Leaflet):', typeof L, L);
                
                // Create tooltip with total time
                const tooltipText = `Route: ${{ids.join(' → ')}}\\nTotal time: ${{totalTime.toFixed(1)}} min`;
                currentRoute = L.polyline(acc, {{ 
                    color:'#1a73e8', 
                    weight:8, 
                    opacity:0.9 
                }}).bindTooltip(tooltipText, {{ sticky: true }});
                
                console.log('Polyline created:', currentRoute);
                currentRoute.addTo(map);
                console.log('Polyline added to map');
                try {{ 
                    const bounds = currentRoute.getBounds();
                    console.log('Bounds:', bounds);
                    map.fitBounds(bounds, {{ padding:[20,20] }}); 
                }} catch(e) {{ console.error('fitBounds error:', e); }}
                status.className = 'ok';
                status.textContent = `Route plotted (${{acc.length}} points, ${{totalTime.toFixed(1)}} min total).`;
                console.log('✓ Route drawn successfully');
            }}

            // Hook up buttons
            window.addEventListener('DOMContentLoaded', () => {{
                const plotBtn = document.getElementById('plotRouteBtn');
                const clearBtn = document.getElementById('clearRouteBtn');
                plotBtn && plotBtn.addEventListener('click', (e) => {{ e.preventDefault(); plotRouteFromInput(); }});
                clearBtn && clearBtn.addEventListener('click', (e) => {{ e.preventDefault(); document.getElementById('routeNodesInput').value=''; document.getElementById('routeStatus').textContent=''; clearRoute(); }});
            }});
        </script>
        """
    m.get_root().html.add_child(folium.Element(control_html))

    m.save(out_html)
    print(f"[OK] Saved map with road-following edges to {out_html}")

###############################################################################
# 7. CLI
###############################################################################

def main():
    if len(sys.argv) < 3:
        print("Usage: python visualize_assignment_folium_roads_knearest.py heritage_assignment_15_time_asymmetric.txt map.osm")
        sys.exit(1)

    assignment_path = sys.argv[1]
    osm_path = sys.argv[2]

    nodes, ways, cameras, meta = parse_assignment_file(assignment_path)
    osm_nodes, graph = load_osm_graph(osm_path)

    print(f"[INFO] Loaded assignment: {len(nodes)} nodes; {len(ways)} ways")
    print(f"[INFO] Loaded OSM graph: {len(osm_nodes)} nodes; "
          f"{sum(len(v) for v in graph.values())} directed edges")

    visualize_with_roads(nodes, ways, cameras, meta, osm_nodes, graph)

if __name__ == "__main__":
    main()
