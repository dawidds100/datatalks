import pickle

with open("pipeline_v1.bin", "rb") as f:
    pipeline = pickle.load(f)

record = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}

data = [record]

prob = pipeline.predict_proba(data)[0, 1]
print(prob)