# Used-Car Price MLOps Project

A complete MLOps cycle (Labs 1–4) on a **used-car price** dataset instead of house prices.

> **Goal:** given a car's age, kilometres driven and engine size, predict its price in **lakh PKR**.

| | House project (teacher) | **Your project** |
|---|---|---|
| Data file | `houses.csv` | `cars.csv` |
| Features | `rooms, size_sqft` | `age_years, mileage_thousand_km, engine_cc` |
| Label | `price_lakh` | `price_lakh` |
| MLflow experiment / model | `house-price-project` / `house-price-model` | `car-price-project` / `car-price-model` |
| Docker image | `house-api` | `car-api` |
| Drift story | 6000 sqft mansions | luxury imports with 3000–4500 cc engines |

**About the data:** `cars.csv` (500 cars) is synthetic, like your teacher's. It follows a hidden rule
(`price ≈ 0.10×cc − 6×age − 0.12×km + 40`, plus noise) so the models have a real pattern to learn. `generate_data.py` shows exactly how it was made.
If you want a real dataset instead (e.g. a Kaggle used-car CSV), see **"Using your own real data"** at the bottom.

---

## 0. What is in this folder

| File | What it does | Lab |
|---|---|---|
| `cars.csv` | The dataset (500 cars) | 1 |
| `data_batches/batch2_new_cars.csv` | 50 more cars ("November data") | 2-B |
| `data_batches/batch3_luxury.csv` | 30 luxury imports (the drift data) | 4 |
| `data_batches/batch4_extra_cars.csv` | 20 more cars (CI retraining experiment) | 2-E13 |
| `add_data.py` | Safely appends a batch to `cars.csv` | 2, 4 |
| `train.py` | Train + MLflow tracking (`linear` / `tree` / `forest`) | 1 |
| `predict.py` | Price one new car using `model.pkl` | 1 |
| `set_champion.py` | *Optional shortcut* for the MLflow UI "champion" step | 1, 2 |
| `export_model.py` | Saves the champion model as `champion.pkl` | 2 |
| `app.py` | FastAPI: `/health`, `/predict`, `/metrics` + prediction diary | 2, 4 |
| `sync_versions.py` | Writes pinned `requirements*.txt`, syncs Python version in Docker + CI | 2 |
| `tests/` | Data tests, quality gate (MAE < 5), API tests | 2 |
| `Dockerfile`, `.dockerignore` | Packs API + champion model | 2 |
| `.github/workflows/ml-pipeline.yml` | CI/CD pipeline | 2 |
| `k8s/deployment.yaml`, `service.yaml` | Kubernetes deployment | 3 |
| `prometheus.yml`, `send_traffic.py`, `check_drift.py` | Monitoring + drift | 4 |

**Rules that never change:** never paste AWS secret keys / Docker tokens into any chat, code or Git.
Always `dvc push` **before** `git push`. Use `python -m dvc` and `python -m mlflow` on Windows.

---

## 1. One-time setup

**Accounts:** AWS, GitHub, Docker Hub (all free). **Programs:** Python 3.11+, Git, AWS CLI v2, Docker Desktop.

Copy this folder to `Desktop\car-price-mlops`, then in PowerShell:

```powershell
cd $HOME\Desktop\car-price-mlops
python --version; git --version; aws --version
python -m pip install pandas scikit-learn joblib mlflow "dvc[s3]" fastapi uvicorn pytest httpx prometheus-client scipy requests
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

---

## 2. LAB 1: Data → tracked, versioned model

### 2.1 Look at the data and train (Labs 1–3 of the guide)
```powershell
Get-Content cars.csv                 # 501 lines = 1 header + 500 cars
python train.py                      # linear
python predict.py                    # ~132.8 lakh for 5 yrs, 60k km, 1300 cc
```
Change the car inside `predict.py` (e.g. 2 years, 20k km, 1800 cc) and run it again.

### 2.2 Compare 3 models with MLflow (Labs 4–6)
```powershell
python train.py
python train.py tree
python train.py forest
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000     # leave this window open
```
Open http://localhost:5000 → **Model training** → **car-price-project** → **Columns** → tick `mae` and `model_type`.

**Expected MAE (with the provided `cars.csv`):**

| Model | MAE | Result |
|---|---|---|
| linear | ~2.70 | Best |
| forest | ~3.67 | Middle |
| tree | ~4.74 | Worst |

Then **Model registry → car-price-model**: add alias **`champion`** to the linear version (lowest MAE).
*Shortcut instead of clicking:* `python set_champion.py` does the same thing.

### 2.3 Prepare AWS S3 (Lab 7)
Do this on the AWS website exactly like the guide (Lab 1 §7.1–7.4 / Lab 2 A1–A5):
bucket `yourname-mlops-dvc-2026` (us-east-1, block public access ON) → IAM user `dvc-user` → policy `dvc-bucket-access`
(the JSON from the guide, bucket name in **both** places) → access key (download the .csv, keep it private) → then:
```powershell
aws configure --profile dvc
aws sts get-caller-identity --profile dvc          # Arn must end in user/dvc-user
aws s3 ls s3://YOUR-BUCKET --profile dvc           # no output = success
aws s3 cp cars.csv s3://YOUR-BUCKET/test.csv --profile dvc
aws s3 rm s3://YOUR-BUCKET/test.csv --profile dvc
```

### 2.4 Version the data with Git + DVC (Lab 8 / Lab 2 Part A)
```powershell
git init -b main
python -m dvc init
python -m dvc config core.analytics false
python -m dvc remote add -d storage s3://YOUR-BUCKET/dvc
python -m dvc remote modify storage region us-east-1
python -m dvc remote modify --local storage profile dvc
python -m dvc add cars.csv
Get-Content cars.csv.dvc                           # md5 = the data's fingerprint
git add .
git status                                         # must NOT list cars.csv or .dvc/config.local
git commit -m "Track cars.csv with DVC, stored in S3"
python -m dvc push
aws s3 ls s3://YOUR-BUCKET --recursive --profile dvc
```
(`.gitignore` is already provided and ignores `mlruns/`, `mlflow.db`, `model.pkl`, `champion.pkl`, `metrics.json`, `predictions.csv`.)

---

## 3. LAB 2: Time travel, API, Docker, CI/CD

### Part B: Time travel with data
```powershell
git log --oneline
(Get-Content cars.csv).Count                       # 501
python add_data.py data_batches/batch2_new_cars.csv     # now 550 cars
python -m dvc status                               # cars.csv: modified
python -m dvc add cars.csv
python -m dvc push
git add cars.csv.dvc
git commit -m "November data: 550 cars"
aws s3 ls s3://YOUR-BUCKET --recursive --profile dvc    # 2 files, nothing overwritten
python train.py                                    # MLflow shows rows_in_dataset = 550, MAE ~2.7
git log --oneline                                  # copy the OLDER commit code, e.g. a8b1531
git checkout a8b1531 cars.csv.dvc
python -m dvc checkout
(Get-Content cars.csv).Count                       # 501 again: October is back
git checkout HEAD cars.csv.dvc
python -m dvc checkout
(Get-Content cars.csv).Count                       # 551 again: November is back
```

### Part C: Model API
```powershell
python set_champion.py             # or set the alias in the MLflow UI; use the lowest-MAE version
python export_model.py             # creates champion.pkl
python -m uvicorn app:app --reload --port 8000
```
Browser: http://localhost:8000/health and http://localhost:8000/docs → POST `/predict`:
```json
{ "age_years": 5, "mileage_thousand_km": 60, "engine_cc": 1300 }
```
Bad inputs must give **422**: `{"age_years": "five", ...}` and `{"age_years": 0, ...}`. Stop with Ctrl+C, then:
```powershell
git add app.py export_model.py set_champion.py predict.py train.py add_data.py generate_data.py
git commit -m "Add car price API (FastAPI) using champion model"
```

### Part D: Docker
Start Docker Desktop first (wait for **Engine running**).
```powershell
python sync_versions.py            # pins YOUR exact library versions + Python version into requirements*.txt, Dockerfile, CI
Get-Content requirements.txt
dir Dockerfile                     # must be "Dockerfile", not "Dockerfile.txt"
docker build -t car-api:model-v6 .          # use YOUR champion version number instead of v6
docker run -d --name car-api -p 8000:8000 car-api:model-v6
Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict -ContentType "application/json" -Body '{"age_years":5,"mileage_thousand_km":60,"engine_cc":1300}'
docker logs car-api
docker exec car-api python -c "import sklearn; print(sklearn.__version__)"
docker stop car-api; docker rm car-api
git add Dockerfile requirements.txt requirements-dev.txt .dockerignore .github
git commit -m "Dockerize car price API"
```

### Part E: CI/CD
```powershell
python train.py
python -m pytest                                   # expect 9 passed (4 data + 1 quality gate + 4 API)
git add .
git status                                         # NOT listed: cars.csv, metrics.json, champion.pkl, .dvc/config.local
git commit -m "Add data, model and API tests"
```
1. GitHub → **New repository** → `car-price-mlops`, **Private**, no README.
2. Push (data first, already done in 2.4):
   ```powershell
   git remote add origin https://github.com/YOUR-NAME/car-price-mlops.git
   git push -u origin main
   ```
3. Make a **second AWS access key** for `dvc-user` (description `github-actions`) and a **Docker Hub token** (Read & Write).
4. Repo → Settings → Secrets and variables → Actions → add **4 secrets**:
   `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`.
5. The pipeline file is already in `.github/workflows/`. Push any change (or Actions → ML pipeline → **Run workflow**) and watch all steps go green.
6. See the score: Actions → run → job → step **Train the model**, or download the **metrics** artifact. See the image: Docker Hub → `car-api` → Tags → `build-1`.

**Experiment 1 – the gate blocks a bad model:** in `tests\test_model.py` set `PASS_MARK = 0.1`, commit and push. The run goes **red at Quality gate** and no image is pushed. Change it back to `5.0` and push again.

**Experiment 2 – new data triggers retraining:**
```powershell
python add_data.py data_batches/batch4_extra_cars.csv
python -m dvc add cars.csv
python -m dvc push
git add cars.csv.dvc
git commit -m "New data: 20 more cars"
git push
```
The pipeline starts by itself, pulls the new data, retrains, and pushes `build-N`.

---

## 4. LAB 3: Kubernetes

1. Docker Desktop → Settings → Kubernetes → **Enable Kubernetes** → wait for the green light.
2. Make sure Docker Hub has **two** tags of `car-api` (run the pipeline twice, e.g. Actions → **Run workflow**) and the repo is **Public**.
3. Edit `k8s\deployment.yaml`: replace `YOUR-DOCKERHUB-NAME` and use the **older** tag (e.g. `build-1`).

```powershell
kubectl config use-context docker-desktop
kubectl config current-context                     # MUST say docker-desktop
kubectl get nodes
kubectl apply -f k8s/
kubectl get pods -w                                # wait for READY 1/1 on both pods (Ctrl+C)
kubectl get service car-api                        # EXTERNAL-IP localhost, port 8080
```
Browser: http://localhost:8080/health and http://localhost:8080/docs.

```powershell
# Self-healing
kubectl get pods
kubectl delete pod PASTE-POD-NAME
kubectl get pods
# Scale
kubectl scale deployment car-api --replicas=3
kubectl get pods
# Release the new model: edit k8s\deployment.yaml to the NEWER tag, then
kubectl apply -f k8s/
kubectl rollout status deployment/car-api
kubectl get deployment car-api -o wide
# Roll back
kubectl rollout undo deployment/car-api
kubectl rollout history deployment/car-api         # then set the YAML tag back to match what is running
# Broken release (readiness probe protects users)
kubectl set image deployment/car-api car-api=YOUR-DOCKERHUB-NAME/car-api:build-999
kubectl get pods                                   # ImagePullBackOff, old pods still Running
kubectl rollout undo deployment/car-api
# Clean up and save
kubectl delete -f k8s/
git add k8s
git commit -m "Add Kubernetes deployment for car price API"
git push
```

---

## 5. LAB 4: Monitoring and drift

**Window 1** (API runs on your laptop; make sure `champion.pkl` exists):
```powershell
cd $HOME\Desktop\car-price-mlops
Remove-Item predictions.csv -ErrorAction SilentlyContinue     # start with a clean diary (tests write to it)
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```
Try `/docs` 2–3 times, then open http://127.0.0.1:8000/metrics and find `predictions_total`.

**Window 2:**
```powershell
cd $HOME\Desktop\car-price-mlops
Get-Content predictions.csv                        # the prediction diary
docker run -d --name prometheus -p 9090:9090 -v "${PWD}\prometheus.yml:/etc/prometheus/prometheus.yml" prom/prometheus:v3.0.0
```
Browser http://localhost:9090 → Status → Target health → `car-api` **UP**. Query, then Graph tab:
```
input_engine_cc_sum / input_engine_cc_count
```
Now simulate users:
```powershell
python send_traffic.py normal
python check_drift.py              # all OK, exit code 0
python send_traffic.py drift       # luxury imports: the model answers confidently but wrongly (silent failure)
python check_drift.py              # DRIFT on all 3 columns
echo $LASTEXITCODE                 # 1
```
Re-run the Prometheus query: the average engine size **jumps** from ~1340 to ~2500.

**Close the loop (retrain):** pretend 30 luxury imports were sold and we now know their real prices.
```powershell
python add_data.py data_batches/batch3_luxury.csv
python train.py                                    # MAE ~4.1: under the pass mark of 5
python -m pytest
python -m dvc add cars.csv
python -m dvc push
git add cars.csv.dvc app.py requirements.txt .gitignore tests/test_api.py send_traffic.py check_drift.py prometheus.yml
git commit -m "Add monitoring, drift check and luxury car data"
git push
```
GitHub Actions retrains, passes the gate and pushes a new `build-N`. **The full circle is complete.**
Clean up: Ctrl+C in window 1, then `docker rm -f prometheus`.

---

## 6. Numbers you should see (provided data, scikit-learn 1.8)

| Moment | Cars | linear MAE |
|---|---|---|
| Start (Lab 1) | 500 | ~2.70 (forest ~3.67, tree ~4.74) |
| After batch 2 (Lab 2-B) | 550 | ~2.68 |
| After extra batch (Experiment 2) | 570 | ~2.55 |
| After luxury batch (Lab 4) | 600 | ~4.10 |

Silent failure demo: the old model underprices the luxury imports by about **22 lakh on average** (for example it answers ~470 lakh where
the real price is ~502 lakh), with no error and a green API. At the luxury step linear (~4.1) is still under the pass mark of 5, while forest (~3.9) and tree (~4.4)
are close; that is normal, and the pipeline always trains `linear` by default. Small differences in your numbers are fine if your library versions differ.

---

## 7. Evidence to collect for your teacher (screenshots)

- [ ] MLflow experiment table with 3 runs (`mae`, `model_type` visible) + registry showing the `champion` alias
- [ ] `cars.csv.dvc` content and the S3 listing (2 data versions)
- [ ] `git log --oneline` + the line count flipping 501 → 551 → 501 during time travel
- [ ] `/docs` page returning a price, and a 422 on bad input
- [ ] `docker images car-api` and the container answering `/predict`
- [ ] Green GitHub Actions run, the **red** run from Experiment 1, and Docker Hub tags
- [ ] `kubectl get pods` (READY 1/1), rollout, rollback, and the build-999 broken release
- [ ] Prometheus graph jumping + `check_drift.py` output (OK, then DRIFT, exit code 1)
- [ ] Final green pipeline after the luxury data

---

## 8. Common problems in this project

| Problem | Fix |
|---|---|
| `champion not found` in `export_model.py` | Set the alias (UI or `python set_champion.py`) |
| Pipeline red at `dvc pull` | Secret names exact? No `profile =` line in `.dvc\config` (only in `config.local`)? Did you `dvc push` before `git push`? |
| Pipeline red at Quality gate | MAE ≥ 5: check the data you added for typos (each row = 4 numbers) |
| `check_drift.py`: "Not enough predictions" | Send traffic first (`send_traffic.py normal`) |
| Drift shows on *normal* traffic | Rare random fluke (about 1 in 250 runs). Run `send_traffic.py normal` again |
| `Dockerfile.txt` / `.yaml.txt` | `ren Dockerfile.txt Dockerfile`; turn on "File name extensions" in Explorer |
| `kubectl` talks to the wrong cluster | `kubectl config use-context docker-desktop` **before every session** |

---

## 9. Using your own real data (optional)

If you prefer a real dataset (e.g. a used-car CSV from Kaggle), keep **numeric columns only**, then change these 5 places consistently:
1. Replace `cars.csv` with your file (label column last, named `price_lakh`, or edit the name everywhere).
2. `train.py` → `FEATURES = [...]` and the printed table in step 5.
3. `app.py` → the `Car` class fields, the `DataFrame` in `/predict`, and `log_prediction`.
4. `predict.py`, `tests/test_data.py` (column list), `tests/test_api.py` (`CAR` dict).
5. `send_traffic.py` and `check_drift.py` column names.

Re-check the MAE first, then set `PASS_MARK` in `tests/test_model.py` a bit above it.

---

## 10. Self-check (viva questions)

1. Feature vs label in this project? Why hide 20% of the cars?
2. Why did the decision tree (MAE ~4.7) lose to linear regression (MAE ~2.7) on this data?
3. What does `cars.csv.dvc` contain, and where is the real data?
4. Why `dvc push` before `git push`? Why is the AWS profile saved with `--local`?
5. Which command brings back October's data, and which brings back the October receipt?
6. Why is `champion.pkl` inside the Docker image but `cars.csv` is not?
7. Which line in `deployment.yaml` decides which model is running? How do you roll back?
8. Why did users not notice the broken `build-999` release?
9. Data drift vs concept drift: luxury imports appear (which?) / all used-car prices rise 20% (which?)
10. What does `sys.exit(1)` in `check_drift.py` allow a pipeline to do?
