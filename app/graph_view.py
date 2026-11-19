"""Interactive citation graph for the Related Work mode (vis-network in the browser).

Nodes: the query, the numbered suggestions (numbers match the list under the graph)
and the similar papers whose citations provide the co-citation evidence. Edges:
query -> suggestion for text matches (dashed), similar paper -> suggestion and
suggestion -> suggestion for citations (arrows).
"""
from __future__ import annotations

import html
import json
from functools import lru_cache
from pathlib import Path

VIS_NETWORK = "https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"
# Bundled copy (Apache-2.0 OR MIT) so the map works offline; the CDN is only a fallback.
VIS_NETWORK_LOCAL = Path(__file__).with_name("static") / "vis-network-9.1.9.min.js"


@lru_cache(maxsize=1)
def vis_script_tag():
    if VIS_NETWORK_LOCAL.exists():
        code = VIS_NETWORK_LOCAL.read_text(encoding="utf-8").replace("</script", "<\/script")
        return f"<script>{code}</script>"
    return f'<script src="{VIS_NETWORK}"></script>'

PALETTE = {
    "light": {"text": "#1f2328", "muted": "#59636e", "query": "#6d28d9", "graph": "#c2410c",
              "text_only": "#1d4ed8", "neighbour": "#9ca3af", "cites": "#c2410c", "cocitation": "#9ca3af",
              "match": "#6d28d9", "panel": "#ffffff"},
    "dark": {"text": "#e6edf3", "muted": "#9da7b3", "query": "#a78bfa", "graph": "#fb923c",
             "text_only": "#60a5fa", "neighbour": "#6b7280", "cites": "#fb923c", "cocitation": "#6b7280",
             "match": "#a78bfa", "panel": "#0e1117"},
}


def _tooltip(*lines):
    return "\n".join(line for line in lines if line)


def build_graph(result, *, theme="dark"):
    """Return (nodes, edges, colors) for vis-network from a related_work() result."""
    c = PALETTE["dark" if theme == "dark" else "light"]
    query = result.get("query", "")
    nodes = [{"id": "__query__", "label": "Your query", "shape": "star", "size": 26,
              "color": c["query"], "title": _tooltip("Query:", query[:300]),
              "font": {"color": c["text"], "size": 15, "face": "sans-serif"}}]
    edges = []
    for number, paper in enumerate(result.get("papers", []), 1):
        ranks = paper.get("source_ranks", {})
        graph_backed = "cocitation" in ranks
        nodes.append({
            "id": paper["id"], "label": str(number), "shape": "dot",
            "size": max(12, 30 - 1.6 * number),
            "color": c["graph"] if graph_backed else c["text_only"],
            "title": _tooltip(f"{number}. {paper['title']} ({paper.get('year') or ''})", f"Why: {paper.get('why', '')}"),
            "font": {"color": c["text"], "size": 16, "face": "sans-serif", "strokeWidth": 0},
        })
        if "bge" in ranks or "bm25" in ranks:
            kinds = [k for k, name in (("bge", "semantic"), ("bm25", "keyword")) if k in ranks]
            edges.append({"from": "__query__", "to": paper["id"], "dashes": True, "width": 1,
                          "color": {"color": c["match"], "opacity": 0.45},
                          "title": " + ".join(dict(bge="semantic", bm25="keyword")[k] for k in kinds) + " match"})
    graph = result.get("graph") or {}
    for node in graph.get("nodes", []):
        nodes.append({"id": node["id"], "label": "", "shape": "dot", "size": 6 + 2 * node.get("cites_suggestions", 1),
                      "color": c["neighbour"],
                      "title": _tooltip(f"Similar paper: {node['title']} ({node.get('year') or ''})",
                                        f"Cites {node.get('cites_suggestions', 0)} of the suggestions")})
    for edge in graph.get("edges", []):
        cites = edge["kind"] == "cites"
        edges.append({"from": edge["source"], "to": edge["target"], "arrows": "to",
                      "width": 2.2 if cites else 1,
                      "color": {"color": c["cites"] if cites else c["cocitation"], "opacity": 0.9 if cites else 0.55},
                      "title": "suggestion cites suggestion" if cites else "similar paper cites this suggestion"})
    return nodes, edges, c


def graph_html(result, *, theme="dark", height=560):
    nodes, edges, c = build_graph(result, theme=theme)
    data = json.dumps({"nodes": nodes, "edges": edges})
    return f"""<!doctype html><html><head><meta charset="utf-8">
{vis_script_tag()}
<style>
  html,body{{margin:0;background:{c['panel']};font-family:sans-serif;color:{c['text']}}}
  #net{{height:{height - 40}px;width:100%}}
  .legend{{display:flex;flex-wrap:wrap;gap:14px;font-size:13px;color:{c['muted']};padding:6px 4px}}
  .dot{{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:5px;vertical-align:-1px}}
  .line{{display:inline-block;width:22px;height:0;border-top:2px solid;margin-right:5px;vertical-align:3px}}
  #fallback{{padding:16px;color:{c['muted']}}}
</style></head><body>
<div class="legend">
  <span><span class="dot" style="background:{c['query']}"></span>query</span>
  <span><span class="dot" style="background:{c['graph']}"></span>suggestion backed by co-citation</span>
  <span><span class="dot" style="background:{c['text_only']}"></span>suggestion from text match only</span>
  <span><span class="dot" style="background:{c['neighbour']}"></span>similar paper that cites suggestions</span>
  <span><span class="line" style="border-color:{c['cites']}"></span>citation between suggestions</span>
  <span><span class="line" style="border-color:{c['match']};border-top-style:dashed"></span>text match</span>
</div>
<div id="net"><div id="fallback">Loading graph… (needs internet access to load the drawing library)</div></div>
<script>
  const data = {data};
  if (window.vis) {{
    document.getElementById("net").innerHTML = "";
    new vis.Network(document.getElementById("net"),
      {{nodes: new vis.DataSet(data.nodes), edges: new vis.DataSet(data.edges)}},
      {{physics: {{solver: "forceAtlas2Based", forceAtlas2Based: {{gravitationalConstant: -60, springLength: 120}},
                  stabilization: {{iterations: 250}}}},
        interaction: {{hover: true, tooltipDelay: 120, navigationButtons: false}},
        edges: {{smooth: {{type: "continuous"}}}}}});
  }} else {{
    document.getElementById("fallback").textContent = "The graph library could not be loaded (offline?). The list below has the same suggestions.";
  }}
</script></body></html>"""
