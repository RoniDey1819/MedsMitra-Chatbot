# MedsMitra Chatbot

MedsMitra is an AI-powered medicine assistant for online pharmacy websites. It combines a Retrieval-Augmented Generation (RAG) backend with a drop-in JavaScript chat widget, so visitors can ask natural-language questions about medicines ("Do you have Paracetamol 500mg?", "What's an alternative to X?") and get answers grounded in real inventory data instead of hallucinated ones.

The repository has three parts:

| Folder                     | What it is                                                                                                                                  |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `backend/`                 | A FastAPI service that embeds a medicine catalog, performs vector similarity search in Supabase (pgvector), and streams answers from an LLM |
| `widget/MedsMitra/`        | A full static demo pharmacy website (HTML/CSS/JS, Bootstrap 5) used to showcase the widget in context                                       |
| `widget/chatbot-widget.js` | The embeddable chat widget itself - a single script that can be dropped into any existing website                                           |

---

## How it works

1. A visitor types a question into the chat widget.
2. The backend rewrites follow-up questions into a standalone query using recent conversation history.
3. The query is embedded locally (`sentence-transformers`) and matched against medicine records in Supabase via a `match_medicines` Postgres function using pgvector cosine similarity.
4. The top matching medicines are passed as context to an LLM (Groq), which is instructed to answer only from that context.
5. The answer is streamed back to the widget over Server-Sent Events (SSE) and rendered in the chat bubble.
6. Conversation history per session is cached in Redis (Upstash) so follow-up questions retain context.

This keeps answers grounded in real stock and dosage data rather than invented information.

---

## Tech stack

**Backend**

- FastAPI + Uvicorn
- Supabase (Postgres + pgvector) for vector storage and similarity search
- `sentence-transformers` (`all-MiniLM-L6-v2`) for local embeddings
- Groq (OpenAI-compatible client) for the LLM response
- Upstash Redis for per-session conversation history
- `trafilatura` / `beautifulsoup4` for optional site crawling (`crawl_site.py`)

**Widget & demo site**

- Vanilla JavaScript (no build step, no framework dependency)
- Bootstrap 5.3 for the demo site layout
- Font Awesome 6.5 icons

---

## Repository structure

```
MedsMitra-Chatbot-main/
├── backend/
│   ├── app.py                 FastAPI app: RAG pipeline + chat endpoint
│   ├── load_data.py           Embeds medicines.csv and upserts into Supabase
│   ├── crawl_site.py          Optional helper to crawl a site for extra context
│   ├── medicines.csv          Sample medicine inventory data
│   ├── supabase_setup.sql     SQL to enable pgvector and create tables/functions
│   └── requirements.txt
├── widget/
│   ├── chatbot-widget.js      Standalone embeddable chat widget
│   ├── frontend.html          Minimal example page using the widget
│   └── MedsMitra/             Full static demo pharmacy website
│       ├── index.html, medicines.html, healthcare.html, labtest.html,
│       │   consultdoctor.html, covidessentials.html, blog.html,
│       │   contact.html, profile.html, addtocart.html
│       ├── css/style.css
│       └── js/app.js
└── README.md
```

---

## Getting started

Quick start for the backend:

```bash
cd backend
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# fill in API keys and Supabase credentials

python load_data.py           # embeds medicines.csv into Supabase
uvicorn app:app --reload --port 8000
```

Then add the widget to any HTML page:

```html
<script>
  window.MED_CHATBOT_API_URL = "http://localhost:8000/chat";
</script>
<script src="/js/chatbot-widget.js"></script>
```

For full setup details - including creating a Groq API key, configuring Supabase, environment variables, and deploying to Render - see the [backend setup guide](README.setup-guide.md).

For the demo pharmacy site, open [`widget/MedsMitra/index.html`](widget/MedsMitra) directly in a browser, or see [`widget/MedsMitra/README.md`](widget/MedsMitra/README.md) for a page-by-page breakdown and design system notes.

---

## Updating medicine data

Edit `backend/medicines.csv` with real inventory (columns: `Medicine_ID, Medicine_Name, Strength, Use_Case, Alternative, Stock, Dosage_Instruction`), then re-run:

```bash
python load_data.py
```

Rows are upserted by `Medicine_ID`, so existing entries are updated in place and new ones are added.

---

## Notes before going live

- Restrict CORS to a real domain via the `ALLOWED_ORIGIN` environment variable.
- Never expose the Supabase `service_role` key in frontend code - it is used server-side only.
- Consider adding rate limiting so a single visitor can't exhaust API credits.
- The demo site in `widget/MedsMitra/` is a static frontend only; its cart, sign-in, and checkout flows are simulated in the browser and do not persist data or call a real backend.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
