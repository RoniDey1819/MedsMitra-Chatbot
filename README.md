# Medical Shop Chatbot — Setup Guide

Two pieces:
- **backend/** — a small Python API that looks up medicines from your CSV (RAG) and asks Grok to write the answer.
- **widget/** — one JS file you paste into your existing HTML pages, which shows a floating chat bubble.

---

## 1. Get a free Grok API key

1. Go to **console.x.ai** and sign up (this is separate from a regular X/grok.com account).
2. Create an API key. New accounts get free starter credits, so you can test without paying.
3. Note the exact model name shown in the console (e.g. `grok-4-fast`) — model names change over time, so use whatever's current there.

---

## 2. Set up the Supabase table

1. Open your Supabase project → **SQL Editor**.
2. Paste in the contents of `backend/supabase_setup.sql` and run it. This:
   - enables the `pgvector` extension
   - creates a `medicines` table (with a `vector(384)` column for embeddings)
   - creates a `match_medicines` function the backend calls to do similarity search
3. Grab two things from **Project Settings → API**:
   - **Project URL** → `SUPABASE_URL`
   - **service_role key** (not the anon key — this runs server-side only) → `SUPABASE_SERVICE_KEY`
4. Grab your **Database → Connection string** (URI, "Session pooler" or direct connection) → `DATABASE_URL`. This is only used by `load_data.py` to write rows directly.

---

## 3. Run the backend locally (test first)

```bash
cd backend
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env: XAI_API_KEY, XAI_MODEL, SUPABASE_URL, SUPABASE_SERVICE_KEY, DATABASE_URL
# leave ALLOWED_ORIGIN=* for now
```

Load your medicine data into Supabase (embeds locally with sentence-transformers, no API cost):
```bash
python load_data.py
```
First run downloads the embedding model (~90MB) and caches it. You'll see the rows getting embedded and upserted.

Now start the API:
```bash
uvicorn app:app --reload --port 8000
```

Test it:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Do you have Paracetamol 500mg?"}'
```
You should get back a JSON reply mentioning stock status, dosage, and alternatives.

---

## 4. Update the medicine data

Edit `backend/medicines.csv` with your real inventory — same columns:
`Medicine_ID, Medicine_Name, Strength, Use_Case, Alternative, Stock, Dosage_Instruction`

Whenever you change the CSV, re-run:
```bash
python load_data.py
```
It upserts by `Medicine_ID`, so existing rows get updated in place and new IDs get added. Data now lives permanently in Supabase — it survives redeploys, unlike the old local ChromaDB file.

---

## 5. Deploy the backend (Render example — Railway is nearly identical)

1. Push the `backend/` folder to a GitHub repo. (`load_data.py` and `supabase_setup.sql` don't need to run on the server — you run those from your own machine whenever data changes. Only `app.py` needs to be live.)
2. On [render.com](https://render.com) → **New + Web Service** → connect the repo.
3. Settings:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
4. Add environment variables (Render dashboard → Environment):
   - `XAI_API_KEY` = your key
   - `XAI_MODEL` = e.g. `grok-4-fast`
   - `ALLOWED_ORIGIN` = your real website URL, e.g. `https://www.yourpharmacy.com`
   - `SUPABASE_URL`, `SUPABASE_SERVICE_KEY` = from Supabase project settings
   - (you do **not** need `DATABASE_URL` on the server — that's only for `load_data.py`, which you run locally)
5. Deploy. Render gives you a URL like `https://medshop-chatbot.onrender.com`.
6. Confirm it's alive: visit `https://medshop-chatbot.onrender.com/health` → should show `{"status":"ok"}`.

**Notes:**
- Free tiers on Render spin down when idle, so the first message after a quiet period can take ~30–50 seconds. Normal, not a bug.
- `sentence-transformers` pulls in PyTorch, which is fairly heavy (roughly 1–2GB installed) and needs more RAM than Render's free 512MB tier typically allows. If the free instance crashes on startup or during embedding, either upgrade to Render's Starter tier (512MB → more RAM), or swap the embedding step for a hosted API (e.g. OpenAI's `text-embedding-3-small`) so the server doesn't need PyTorch at all — ask if you want that version instead.

---

## 6. Add the chatbot to your website

Upload `widget/chatbot-widget.js` to your website's files (anywhere, e.g. `/js/chatbot-widget.js`), then add this near the end of `<body>` on every page you want the chat bubble to appear:

```html
<script>
  window.MED_CHATBOT_API_URL = "https://medshop-chatbot.onrender.com/chat";
</script>
<script src="/js/chatbot-widget.js"></script>
```

That's it — a chat bubble appears bottom-right. No build tools, no framework needed since your site is plain HTML/CSS/JS.

---

## 7. How it actually answers questions (the RAG part)

1. User types a question in the widget → sent to your backend's `/chat` endpoint.
2. The backend embeds the question locally (same `all-MiniLM-L6-v2` model used to load the data) and calls the `match_medicines` Postgres function in Supabase via RPC, which does a cosine-similarity search over the `embedding` column using pgvector.
3. The top 3 matching medicine rows come back as "context."
4. That context + the user's question go to Grok, with instructions to answer **only** from that context and never invent stock/dosage info.
5. Grok's answer is sent back and shown in the widget.

This means the bot won't hallucinate medicines you don't stock — it's grounded in your actual Supabase data, and that data now persists properly (redeploying the backend no longer wipes it, since it's no longer in a local file on disk).

---

## 8. Things worth doing before going live

- **Restrict CORS**: set `ALLOWED_ORIGIN` to your real domain (not `*`) so random sites can't call your API and burn your Grok credits.
- **Rate limiting**: consider adding basic rate limiting (e.g. `slowapi`) so one visitor can't spam requests.
- **Medical disclaimer**: the system prompt already tells the bot to recommend confirming with a pharmacist — keep that, since this is guidance, not medical advice.
- **Row Level Security**: the backend uses the Supabase **service_role** key, which bypasses RLS — that's fine since it only runs server-side and is never exposed to the browser. Never put the service_role key in the widget or any frontend code.
- **Bigger inventory**: pgvector with an `ivfflat` index (as set up in `supabase_setup.sql`) comfortably handles tens of thousands of rows. If you grow past that, revisit the index type (e.g. `hnsw`) and Supabase's compute tier.
