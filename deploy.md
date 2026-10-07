# Deploying

The API and the Streamlit app are deployed separately, from the same GitHub repository. Both are free on the hosts below. Neither needs the other: each one loads its own copy of the model.

The model file is not in Git (`artifacts/` is gitignored). The API host builds it during the build step. Streamlit Cloud has no build step, so the app trains the model on the first prediction after it starts, which takes about a minute.

Python 3.11 or newer is required by the pinned packages.

## 1. API on Render

1. Push the repository to GitHub.
2. Sign in at https://render.com with GitHub and click **New > Web Service**.
3. Pick the `House-Price-Prediction-API` repository and fill in:

   | Setting | Value |
   |---------|-------|
   | Branch | `main` |
   | Runtime | Python 3 |
   | Build command | `pip install -r requirements.txt && python train.py` |
   | Start command | `uvicorn app.main:app --host 0.0.0.0 --port $PORT` |
   | Instance type | Free |

4. Under **Environment**, add `PYTHON_VERSION` = `3.13` so Render doesn't pick an older Python.
5. Click **Create Web Service**. The build log ends with `Model saved to .../house_price_model.pkl`.
6. Check it: open `https://<your-service>.onrender.com/docs` and run `POST /predict` with the sample body.

Share the `/docs` URL with anyone who wants to call the API. CORS allows every origin, so browser apps on any site can call it.

Free Render services sleep after 15 minutes without traffic. The first request after that takes up to a minute while the service wakes up.

## 2. Streamlit app on Streamlit Community Cloud

1. Sign in at https://share.streamlit.io with GitHub and click **Create app**.
2. Fill in:

   | Setting | Value |
   |---------|-------|
   | Repository | `TusharParlikar/House-Price-Prediction-API` |
   | Branch | `main` |
   | Main file path | `streamlit_app.py` |

3. Open **Advanced settings** and pick Python 3.13 (or any 3.11+).
4. Click **Deploy**. Streamlit installs `requirements.txt` from the repository root.
5. The first **Predict price** click downloads the dataset and trains the model, which takes about a minute. Later predictions are instant until the app restarts.

## 3. Test page (optional)

`test/index.html` is a static page. To test the deployed API with it:

- Open `test/index.html` in a browser, set **API URL** to `https://<your-service>.onrender.com`, and click **Predict price**.
- Or host it: drag the `test/` folder onto https://app.netlify.com/drop to get a public URL.

## Updating

Push to `main`. Render and Streamlit Cloud both redeploy automatically, and both retrain the model as part of the new deploy.

## Running on your own server

```bash
pip install -r requirements.txt
python train.py                    # train once, before starting workers
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

Train before starting several workers, or every worker trains its own model at the same time.
