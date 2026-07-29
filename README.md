# Researcher AI — Engineering Codes & Standards

A serverless, mobile-optimized web application powered by **Google Gemini AI** and **GitHub Pages**. Designed for engineers to search, query, and chat with engineering codes, standards, specs, and datasheets on the fly from any smartphone or desktop browser.

---

## 🌟 Key Features

- 📱 **Mobile-First & Serverless**: Hosted on **GitHub Pages** for free. Works directly on any phone or desktop browser with zero server setup or hosting fees.
- ⚙️ **Powered by Google Gemini**: Connects directly to `gemini-2.5-flash` / `gemini-1.5-flash` for high-speed technical Q&A, and `text-embedding-004` for semantic search.
- ⚡ **Dual Document Management**:
  1. **GitHub Repository Library (`codes/`)**: Place PDFs in the `codes/` directory. GitHub Actions automatically parses, chunks, and indexes them using Python on GitHub's cloud servers.
  2. **Direct Mobile Uploads**: Upload PDFs, Word (`.docx`), or TXT files directly on your phone via the web app interface. Indexed locally on your device in **IndexedDB**.
- 🔍 **Sub-Millisecond Hybrid RAG**: BM25 keyword matching (for clause numbers like *"ASME B31.3 Para 304.1.2"*) fused with vector semantic embeddings.
- 📌 **Exact Source Citations**: Every answer cites file names, section/clause identifiers, page numbers, and expandable excerpt cards.
- 💾 **Export & Import Backup**: Sync your PC and phone by exporting an offline `.json` database backup with 1 click.

---

## 🚀 Quick Setup & Deployment to GitHub Pages

### 1. Push to GitHub & Enable Pages
1. Push this codebase to your GitHub repository.
2. In your GitHub repository, go to **Settings** → **Pages**.
3. Under **Build and deployment** → **Source**, select **GitHub Actions**.

### 2. Add your Gemini API Key Secret
1. Go to your GitHub Repository **Settings** → **Secrets and variables** → **Actions**.
2. Click **New repository secret**.
3. Name: `GEMINI_API_KEY`
4. Value: Paste your Google Gemini API Key (get one free at [aistudio.google.com](https://aistudio.google.com/)).

### 3. Add Engineering Codes & Standards
1. Place your PDF/Word files inside the `codes/` folder in your repository (e.g. `codes/ASME_B31.3.pdf`).
2. Commit and push:
   ```bash
   git add codes/
   git commit -m "Add engineering codes"
   git push origin main
   ```
3. GitHub Actions will automatically wake up, run the Python indexing pipeline, compute Gemini embeddings, and deploy your live app to GitHub Pages!

---

## 📱 Using on your Mobile Phone

1. Open your GitHub Pages URL on your smartphone (e.g., `https://<your-username>.github.io/researcher`).
2. Tap ⚙️ **Settings** and enter your Gemini API Key (saved locally on your phone).
3. Search or ask questions regarding any engineering code clause or calculation!

---

## 🛠 Local Development & Testing

To test locally on your computer:

```bash
# 1. Run the Python indexing script (Optional: set GEMINI_API_KEY environment variable)
python scripts/build_index.py

# 2. Start a local HTTP server
python -m http.server 8000
```
Open `http://localhost:8000` in your browser.
