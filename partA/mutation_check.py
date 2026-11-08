"""Breaks eta_app.py in 8 ways and checks the test suite catches each one (Part A2)."""
import shutil, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = (HERE / "eta_app.py").read_text()

MUTATIONS = {
    "no cross-field rule": [("if self.distance_km > 150 and self.n_stops > 20:", "if False:")],
    "no rounding": [("preds = [round(float(v), 1) for v in y]", "preds = [float(v) for v in y]")],
    "batch limit 1000": [("max_length=500", "max_length=1000")],
    "/metrics counted": [('if request.url.path != "/metrics":', "if True:")],
    "500 leaks internals": [("    except Exception:\n        log.exception(\"inference failed",
                             "    except Exception as e:\n        log.exception(\"inference failed"),
                            ('content={"error": "internal_error", "request_id": rid})\n    return {',
                             'content={"error": repr(e), "request_id": rid})\n    return {')],
    "request id not echoed": [("rid = incoming if 0 < len(incoming) <= 128 else str(uuid.uuid4())",
                               "rid = str(uuid.uuid4())")],
    "batch order reversed": [("preds = [round(float(v), 1) for v in y]",
                              "preds = [round(float(v), 1) for v in y][::-1]")],
    "weight_kg >= 0 allowed": [("weight_kg: float = Field(gt=0, le=1000)", "weight_kg: float = Field(ge=0, le=1000)")],
}

def run_suite(app_src: str) -> tuple[int, str]:
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "eta_app.py").write_text(app_src)
        shutil.copy(HERE / "test_eta_app.py", d)
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", d],
                           capture_output=True, text=True, cwd=HERE.parent)
        return r.returncode, r.stdout.strip().splitlines()[-1]

def main() -> int:
    code, summary = run_suite(SRC)
    print(f"{'unmodified eta_app.py':26s} -> {summary}")
    assert code == 0, "the suite must pass on the real app first"
    killed = 0
    for name, edits in MUTATIONS.items():
        src = SRC
        for old, new in edits:
            assert old in src, f"mutation {name!r}: pattern not found"
            src = src.replace(old, new)
        code, summary = run_suite(src)
        killed += code != 0
        print(f"{name:26s} -> {'KILLED  ' if code else 'SURVIVED'} ({summary})")
    print(f"\n{killed}/{len(MUTATIONS)} mutations killed")
    return 0 if killed == len(MUTATIONS) else 1

if __name__ == "__main__":
    sys.exit(main())
