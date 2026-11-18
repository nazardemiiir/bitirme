"""Manhattan yol ağını OpenStreetMap'ten indirir (yalnızca bir kez çalıştırılır).

Terminalde: python prepare_roadnet.py
İnternet hızına göre 1-3 dakika sürebilir.
"""
from ml.roadnet import GRAPH_PATH, download_graph

if __name__ == "__main__":
    print("Manhattan yol ağı indiriliyor...")
    G = download_graph()
    print(f"Tamam: {len(G.nodes):,} kavşak, {len(G.edges):,} yol parçası")
    print(f"Kaydedildi: {GRAPH_PATH}")
