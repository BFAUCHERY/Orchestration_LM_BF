

#!/usr/bin/env python3
"""
stress_test.py
"""
# Envoie N requêtes concurrentes à l'endpoint /predict d'une application Flask déployée sur Kubernetes.

import argparse
import concurrent.futures
import time
import requests
from io import BytesIO
from PIL import Image

# ------------------------------
# Configuration à adapter
# ------------------------------
URL = "http://127.0.0.1:55013/predict"  # Ex. obtenu via `minikube service --url`
IMAGE_PATH = "test_image.jpg"
NUM_REQUESTS = 100
MAX_WORKERS = 10
# ------------------------------

def send_request(_):
    """Charge l'image et poste vers l'API."""
    with Image.open(IMAGE_PATH) as img:
        buf = BytesIO()
        img.save(buf, format='JPEG')
        buf.seek(0)
        files = {'file': ('test_image.jpg', buf, 'image/jpeg')}
        start = time.time()
        resp = requests.post(URL, files=files)
        latency = time.time() - start
        return resp.status_code, latency

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default=URL, help="URL de l'API /predict")
    parser.add_argument('--requests', type=int, default=NUM_REQUESTS, help="Nombre total de requêtes")
    parser.add_argument('--workers', type=int, default=MAX_WORKERS, help="Nombre de threads")
    args = parser.parse_args()

    latencies = []
    statuses = []

    print(f"Lancement de {args.requests} requêtes vers {args.url} avec {args.workers} workers…")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as exe:
        futures = [exe.submit(send_request, i) for i in range(args.requests)]
        for f in concurrent.futures.as_completed(futures):
            status, lat = f.result()
            statuses.append(status)
            latencies.append(lat)

    print("Statuts reçus:", set(statuses))
    print(f"Latence moyenne : {sum(latencies)/len(latencies):.3f}s")
    print(f"Latence p95 : {sorted(latencies)[int(0.95*len(latencies))-1]:.3f}s")

if __name__ == "__main__":
    main()