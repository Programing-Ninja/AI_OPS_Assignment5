"""Deploys eta_ray on a 2-CPU Ray and checks (a)-(e) of Part C2."""
import sys, time, json, argparse, collections, threading
from pathlib import Path

import pandas as pd
import requests
import ray
from ray import serve

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import eta_ray  # noqa: E402

PORT = 8100
URL = f"http://127.0.0.1:{PORT}/v1/models/eta:predict"
APP = "eta"
FEATURES = ["distance_km", "n_stops", "hour", "weekday", "traffic_index", "rain", "weight_kg"]
RESULTS = []

def check(name, cond, evidence):
    RESULTS.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}\n       {evidence}")
    assert cond, name

def deploy(weight, extra_work_ms=0.0):
    t0 = time.time()
    serve.run(eta_ray.build(weight=weight, extra_work_ms=extra_work_ms), name=APP, route_prefix="/")
    return round(time.time() - t0, 1)

def replicas():
    st = serve.status().applications[APP]
    return {n: dict(d.replica_states) for n, d in st.deployments.items()}

def route_all(sess, ids, rows, headers=None):
    """Sends one row per customer and returns {customer_id: (arm, version, replica)}."""
    out = {}
    for i, cid in enumerate(ids):
        r = sess.post(URL, json={"customer_id": cid, "instances": [rows[i % len(rows)]]},
                      headers=headers or {}, timeout=10)
        r.raise_for_status()
        j = r.json()
        assert j["model_version"] == j["arm"], f"arm/version mismatch for {cid}: {j}"
        out[cid] = (j["arm"], j["model_version"], j["served_by"])
    return out

def autoscale_demo(sess, rows):
    print("\n--- optional: autoscaling demo (candidate gets 100 % of traffic, 40 ms of simulated work) ---")
    deploy(weight=1.0, extra_work_ms=40)
    stop, timeline, t0 = threading.Event(), [], time.time()

    def poll():
        while not stop.is_set():
            n = replicas().get("EtaModelCandidate", {}).get("RUNNING", 0)
            timeline.append((round(time.time() - t0), n)); time.sleep(1)

    th = threading.Thread(target=poll); th.start()

    def hammer(k):
        s = requests.Session(); end = time.time() + 25
        while time.time() < end:
            s.post(URL, json={"customer_id": f"load-{k}", "instances": [rows[k]]}, timeout=30)

    ts = [threading.Thread(target=hammer, args=(k,)) for k in range(8)]
    [t.start() for t in ts]; [t.join() for t in ts]
    time.sleep(5); stop.set(); th.join()
    print("   (t_s, candidate RUNNING replicas):", timeline[::3])
    print(f"   max replicas reached: {max(n for _, n in timeline)} (max_replicas="
          f"{eta_ray.CANDIDATE_AUTOSCALING['max_replicas']})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--autoscale-demo", action="store_true")
    a = ap.parse_args()

    rows = pd.read_csv(ROOT / "data" / "eta_fresh.csv")[FEATURES].head(200).to_dict(orient="records")
    ray.init(num_cpus=2, include_dashboard=False, logging_level="ERROR", log_to_driver=False,
             object_store_memory=100 * 1024 ** 2,
             runtime_env={"env_vars": {"PYTHONPATH": str(HERE), "PYTHONNOUSERSITE": "1"}})
    serve.start(http_options={"host": "127.0.0.1", "port": PORT}, logging_config={"log_level": "WARNING"})
    sess = requests.Session()
    try:
        print("=== setup")
        print(f"Ray cluster resources: CPU={ray.cluster_resources().get('CPU')}")
        budget = (eta_ray.ROUTER_CPUS + eta_ray.PROD_REPLICAS * eta_ray.MODEL_CPUS
                  + eta_ray.CANDIDATE_AUTOSCALING["max_replicas"] * eta_ray.MODEL_CPUS)
        print(f"worst-case CPU reservation: router {eta_ray.ROUTER_CPUS} + prod {eta_ray.PROD_REPLICAS}x"
              f"{eta_ray.MODEL_CPUS} + candidate {eta_ray.CANDIDATE_AUTOSCALING['max_replicas']}x"
              f"{eta_ray.MODEL_CPUS} = {budget:.2f} CPU (<= 2)")
        assert budget <= 2.0
        print("candidate autoscaling_config:", json.dumps(eta_ray.CANDIDATE_AUTOSCALING))
        print(f"deployed weight=0.2 in {deploy(0.2)} s; replicas: {replicas()}")
        print(f"available CPU after deploy: {ray.available_resources().get('CPU'):.2f}")

        ids = [f"cust-{i:05d}" for i in range(1000)]
        print("\n=== (a) share of 1000 distinct customers routed to the candidate")
        first = route_all(sess, ids, rows)
        share = sum(arm == "v2" for arm, _, _ in first.values()) / len(ids)
        local = sum(eta_ray.choose_arm(c, 0.2) == "v2" for c in ids) / len(ids)
        check("(a) candidate share within 0.2 +- 0.04", abs(share - 0.2) <= 0.04,
              f"share = {share:.3f} ({int(share * 1000)} of 1000); offline bucket() predicts {local:.3f}; "
              f"prod served by {len({s for a_, _, s in first.values() if a_ == 'v1'})} replicas")

        print("\n=== (b) stickiness: 200 customers call a second time (with a different payload row)")
        sample = ids[:200]
        second = route_all(sess, sample, rows[::-1])
        same = sum(first[c][0] == second[c][0] for c in sample)
        check("(b) 200/200 customers keep their arm", same == 200,
              f"{same}/200 identical; of these {sum(second[c][0] == 'v2' for c in sample)} on v2")

        print("\n=== (c) delete + redeploy the identical app (new replicas, new processes)")
        old_replicas = {s for _, _, s in first.values()}
        serve.delete(APP); time.sleep(2)
        print(f"   redeployed in {deploy(0.2)} s; replicas: {replicas()}")
        third = route_all(sess, sample, rows)
        new_replicas = {s for _, _, s in third.values()}
        same = sum(first[c][0] == third[c][0] for c in sample)
        check("(c) 200/200 customers keep their arm after redeploy", same == 200,
              f"{same}/200 identical; replica ids overlap with the first deployment: "
              f"{len(old_replicas & new_replicas)} (0 = genuinely new replicas)")

        print("\n=== (d) ramp the canary 0.2 -> 0.5")
        print(f"   updated in {deploy(0.5)} s; replicas: {replicas()}")
        ramp = route_all(sess, ids, rows)
        was_v2 = [c for c in ids if first[c][0] == "v2"]
        moved_back = [c for c in was_v2 if ramp[c][0] != "v2"]
        new_share = sum(arm == "v2" for arm, _, _ in ramp.values()) / len(ids)
        newly = sum(first[c][0] == "v1" and ramp[c][0] == "v2" for c in ids)
        check("(d) nobody on the candidate moves back to prod", not moved_back,
              f"{len(was_v2)} customers were on v2 -> {len(was_v2) - len(moved_back)} still on v2, "
              f"{len(moved_back)} moved back; {newly} prod customers newly joined v2; new share {new_share:.3f}")

        print("\n=== (e) X-Beta: 1 header override for 50 prod customers")
        prod_customers = [c for c in ids if ramp[c][0] == "v1"][:50]
        beta = route_all(sess, prod_customers, rows, headers={"X-Beta": "1"})
        forced = sum(arm == "v2" for arm, _, _ in beta.values())
        control = route_all(sess, prod_customers[:10], rows, headers={"X-Beta": "0"})
        check("(e) X-Beta: 1 forces the candidate on 50/50 requests", forced == 50 and len(beta) == 50,
              f"{forced}/50 routed to v2 with the header; same customers without it (X-Beta: 0): "
              f"{collections.Counter(a_ for a_, _, _ in control.values())}")

        r = sess.post(URL, json={"customer_id": "c1", "instances": [{**rows[0], "hour": 25}]})
        print(f"\n(extra) invalid input through the router -> HTTP {r.status_code}")

        if a.autoscale_demo:
            autoscale_demo(sess, rows)
    finally:
        serve.shutdown(); ray.shutdown()

    print(f"\nALL {len(RESULTS)} CHECKS PASSED" if all(ok for _, ok in RESULTS) else "SOME CHECKS FAILED")

if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"ASSERTION FAILED: {e}")
        sys.exit(1)
