import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
samples = ROOT / "api" / "sample_requests"
legit = json.loads((samples / "legit.json").read_text())
fraud = json.loads((samples / "fraud.json").read_text())


def item(name, method, path, body=None, tests=()):
    request = {"method": method, "header": [], "url": "{{baseUrl}}" + path}
    if body is not None:
        request["header"] = [{"key": "Content-Type", "value": "application/json"}]
        request["body"] = {"mode": "raw", "raw": json.dumps(body, indent=2)}
    return {
        "name": name,
        "event": [{"listen": "test", "script": {"type": "text/javascript", "exec": list(tests)}}],
        "request": request,
    }


def status(code):
    return f'pm.test("status is {code}", function () {{ pm.response.to.have.status({code}); }});'


predict_tests = [
    status(200),
    'pm.test("probability is between 0 and 1", function () { var p = pm.response.json().fraud_probability; pm.expect(p).to.be.within(0, 1); });',
    'pm.test("decision matches threshold", function () { var b = pm.response.json(); pm.expect(b.decision === "flag_for_review").to.eql(b.fraud_probability >= b.threshold); });',
    'pm.test("states it is a risk signal", function () { pm.expect(pm.response.json().note).to.include("not proof of fraud"); });',
]

missing = copy.deepcopy(legit)
del missing["features"]["V5"]
negative = copy.deepcopy(legit)
negative["features"]["Amount"] = -5
non_numeric = copy.deepcopy(legit)
non_numeric["features"]["V1"] = "abc"

collection = {
    "info": {
        "name": "Fraud Risk API",
        "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json",
    },
    "variable": [{"key": "baseUrl", "value": "http://127.0.0.1:8000"}],
    "item": [
        item("Health check", "GET", "/health", tests=[
            status(200),
            'pm.test("model is loaded", function () { pm.expect(pm.response.json().model_loaded).to.eql(true); });',
        ]),
        item("Predict: legitimate sample", "POST", "/predict", legit, predict_tests),
        item("Predict: fraud sample", "POST", "/predict", fraud, predict_tests),
        item("Invalid: missing V5 (expects 422)", "POST", "/predict", missing, [status(422)]),
        item("Invalid: negative amount (expects 422)", "POST", "/predict", negative, [status(422)]),
        item("Invalid: non-numeric V1 (expects 422)", "POST", "/predict", non_numeric, [status(422)]),
        item("Recent transactions", "GET", "/transactions?limit=5", tests=[
            status(200),
            'pm.test("returns a list", function () { pm.expect(pm.response.json().transactions).to.be.an("array"); });',
        ]),
        item("Operational metrics", "GET", "/metrics", tests=[
            status(200),
            'pm.test("reports database availability", function () { pm.expect(pm.response.json()).to.have.property("database_available"); });',
        ]),
    ],
}

out = ROOT / "api" / "postman_collection.json"
out.write_text(json.dumps(collection, indent=2))
print("Wrote", out)