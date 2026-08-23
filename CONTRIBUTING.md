# Contributing to MedsMitra Chatbot

Thanks for your interest in improving MedsMitra. This document covers how to set up the project locally, the conventions used across the repo, and how to submit changes.

---

## Project structure

Before contributing, it helps to know where things live:

- `backend/` - FastAPI RAG service (Python)
- `widget/chatbot-widget.js` - the embeddable chat widget (vanilla JS)
- `widget/MedsMitra/` - static demo pharmacy site (HTML/CSS/JS)

See the main [README.md](README.md) for a full architecture overview.

---

## Getting started

1. Fork the repository and clone your fork:

   ```bash
   git clone https://github.com/<your-username>/MedsMitra-Chatbot.git
   cd MedsMitra-Chatbot
   ```

2. Create a branch for your change:

   ```bash
   git checkout -b feature/short-description
   ```

3. Set up the backend locally:

   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   cp .env.example .env
   # fill in your own API keys and Supabase credentials - never commit .env
   ```

4. For widget or demo-site changes, no build step is needed - just open the relevant HTML file in a browser or use a local static server (e.g. VS Code Live Server).

---

## Making changes

- Keep pull requests focused on a single change (one feature, one fix, one refactor) rather than bundling unrelated changes together.
- Match the existing code style in the file you're editing rather than introducing a new style.
- For backend changes, run the app locally and confirm `/chat` and `/health` still respond correctly before submitting.
- For widget changes, test in a plain HTML page with `MED_CHATBOT_API_URL` pointed at a local backend.
- For demo-site changes, check that the page still renders correctly across a few pages (`index.html`, `medicines.html`, etc.), since shared styles live in `css/style.css` and shared scripts in `js/app.js`.
- Do not commit secrets, API keys, or `.env` files. `backend/.gitignore` already excludes `.env` and `venv/` - don't remove those entries.
- If you update `backend/medicines.csv`, mention it in your PR description, since reviewers will want to re-run `load_data.py` against a test Supabase instance to verify it loads cleanly.

---

## Commit messages

Use short, descriptive commit messages in the imperative mood, e.g.:

```
Fix CORS handling for missing ALLOWED_ORIGIN
Add rate limiting to /chat endpoint
Update medicines.csv with new inventory
```

---

## Submitting a pull request

1. Push your branch to your fork.
2. Open a pull request against the `main` branch of this repository.
3. In the PR description, include:
   - What the change does and why
   - How you tested it (e.g. "ran locally, tested `/chat` with sample queries")
   - Any new environment variables or setup steps required
4. Link any related issues.

A maintainer will review your PR and may request changes before merging.

---

## Reporting bugs

When opening an issue, please include:

- Steps to reproduce the problem
- What you expected to happen vs. what actually happened
- Relevant logs or error messages (with any API keys/secrets redacted)
- Whether the issue is in the backend, the widget, or the demo site

---

## Suggesting features

Feature requests are welcome. Open an issue describing:

- The problem you're trying to solve
- Your proposed approach, if you have one
- Whether you're able to help implement it

---

## Code of conduct

Be respectful and constructive in issues, discussions, and pull requests. Assume good intent, and keep feedback focused on the code and ideas rather than the person.
