"""Release gate: exits 0 if all checks pass, 1 otherwise (Part D)."""
import sys, math, time, argparse
import requests

PREDICT = "/v1/models/eta:predict"
BASE_ROW = dict(distance_km=30.0, n_stops=6, hour=17, weekday=2, traffic_index=0.5, rain=0, weight_kg=8.0)

class Smoke:
    def __init__(self, a):
        self.base = a.url.rstrip("/")
        if self.base.endswith(PREDICT):
            self.base = self.base[: -len(PREDICT)]
        self.a, self.sess, self.results = a, requests.Session(), []
        self.headers = {"X-Beta": "1"} if a.beta else {}

    def body(self, rows):
        b = {"instances": rows}
        if self.a.customer_id:
            b["customer_id"] = self.a.customer_id
        return b

    def predict(self, rows, headers=None):
        return self.sess.post(self.base + PREDICT, json=self.body(rows), timeout=self.a.timeout,
                              headers={**self.headers, **(headers or {})})

    def preds(self, rows):
        r = self.predict(rows)
        r.raise_for_status()
        return r.json()["predictions"]

    def check(self, name, fn):
        try:
            ok, detail = fn()
        except requests.ConnectionError:
            raise
        except Exception as e:
            ok, detail = False, f"{type(e).__name__}: {e}"
        self.results.append((name, ok, detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name:28s} {detail}")

    def monotone(self, field, values):
        ys = self.preds([{**BASE_ROW, field: v} for v in values])
        ok = all(b >= a for a, b in zip(ys, ys[1:]))
        return ok, f"{field} {values} -> ETA {ys}"

    def run(self):
        a = self.a

        def health():
            r = self.sess.get(self.base + "/healthz", timeout=a.timeout)
            j = r.json()
            return r.status_code == 200 and j.get("status") == "ok", f"HTTP {r.status_code} {j}"

        def contract():
            rows = [BASE_ROW, {**BASE_ROW, "distance_km": 5}, {**BASE_ROW, "distance_km": 90}]
            r = self.predict(rows, headers={"X-Request-ID": "smoke-contract"})
            j = r.json()
            ok = (r.status_code == 200 and len(j["predictions"]) == 3
                  and all(isinstance(p, (int, float)) and math.isfinite(p) for p in j["predictions"])
                  and all(round(p, 1) == p for p in j["predictions"])
                  and r.headers.get("X-Request-ID") == "smoke-contract")
            return ok, f"HTTP {r.status_code}, 3 rows -> {j.get('predictions')}, request id echoed"

        def version():
            v = self.predict([BASE_ROW]).json().get("model_version")
            return v == a.expect_version, f"served {v!r}, expected {a.expect_version!r}"

        def plausible():
            ys = self.preds([{**BASE_ROW, "distance_km": d, "n_stops": s} for d, s in [(0.5, 1), (30, 6), (200, 20)]])
            return all(1 <= y <= 24 * 60 for y in ys), f"ETAs {ys} within [1, 1440] min"

        def deterministic():
            y1, y2 = self.preds([BASE_ROW]), self.preds([BASE_ROW])
            return y1 == y2, f"{y1} == {y2}"

        def invalid_422():
            r = self.predict([{**BASE_ROW, "hour": 25}])
            return r.status_code == 422, f"hour=25 -> HTTP {r.status_code}"

        def infeasible_422():
            r = self.predict([{**BASE_ROW, "distance_km": 200, "n_stops": 30}])
            return r.status_code == 422 and "infeasible" in r.text, f"200 km & 30 stops -> HTTP {r.status_code}"

        def latency():
            for _ in range(5):
                self.predict([BASE_ROW])
            ts = []
            for i in range(a.n_latency):
                t0 = time.perf_counter()
                r = self.predict([{**BASE_ROW, "distance_km": 5 + i}])
                ts.append((time.perf_counter() - t0) * 1000)
                if r.status_code != 200:
                    return False, f"request {i} failed with HTTP {r.status_code}"
            ts.sort()
            p95 = ts[min(len(ts) - 1, int(0.95 * len(ts)))]
            return p95 <= a.max_p95_ms, f"p95 {p95:.1f} ms over {a.n_latency} requests (limit {a.max_p95_ms} ms)"

        self.check("health endpoint", health)
        self.check("response contract", contract)
        self.check("model version", version)
        self.check("plausible range", plausible)
        self.check("deterministic", deterministic)
        self.check("sanity: distance monotone", lambda: self.monotone("distance_km", [2, 10, 30, 60, 120]))
        self.check("sanity: traffic monotone", lambda: self.monotone("traffic_index", [0.0, 0.25, 0.5, 0.75, 1.0]))
        self.check("sanity: stops monotone", lambda: self.monotone("n_stops", [1, 5, 10, 20]))
        self.check("invalid input -> 422", invalid_422)
        self.check("cross-field rule -> 422", infeasible_422)
        self.check("latency p95", latency)

        failed = [n for n, ok, _ in self.results if not ok]
        print(f"\n{len(self.results) - len(failed)}/{len(self.results)} checks passed -> "
              + ("RELEASE OK (exit 0)" if not failed else f"BLOCK RELEASE (exit 1): {', '.join(failed)}"))
        return 0 if not failed else 1

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--expect-version", required=True)
    ap.add_argument("--max-p95-ms", type=float, default=80.0)
    ap.add_argument("--customer-id", help="send customer_id (needed by the Ray Serve router)")
    ap.add_argument("--beta", action="store_true", help="send X-Beta: 1 (force the canary arm)")
    ap.add_argument("--n-latency", type=int, default=50)
    ap.add_argument("--timeout", type=float, default=5.0)
    a = ap.parse_args(argv)
    try:
        return Smoke(a).run()
    except requests.ConnectionError as e:
        print(f"[FAIL] cannot reach {a.url} ({e.__class__.__name__}) -> BLOCK RELEASE (exit 1)")
        return 1

if __name__ == "__main__":
    sys.exit(main())
