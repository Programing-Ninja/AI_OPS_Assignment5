

Claude Code (terminal).

How was it used :
- Claude drafted the boilerplate and helped in writting where I was stuck in the FastAPI service, the tests, the MLflow pyfunc and
  gate, the Ray Serve sticky canary, the verification script and the tests. It also helped in the helper
  scripts that run the load tests and the shadow comparison. I reviewed the code, ran it on my laptop, and can
  explain every part of it.
-  Claude was used to diagnose environment problems, including:
  - a 44 ms Nagle/delayed-ACK stall with `uvicorn --workers 2` when httptools/uvloop are missing
    (`partA/nagle_note.txt`);
  - `serve run` needing `--working-dir`;
  - a ROS `PYTHONPATH` that broke pytest.
- Claude helped in structure of the report.

Every number, table and output file comes from actually running the submitted
code on my machine. No metric was invented or edited by hand.

AI assistance sped up writing the code and finding the root cause of performance anomalies. The
results come from real runs. I understand the design choices and can defend them in a whiteboard test
