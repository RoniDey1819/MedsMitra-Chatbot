# 💊 MedsMitra – Health & Wellness Web Platform (Frontend)

MedsMitra is a responsive, multi-page frontend website built with HTML, CSS and Bootstrap 5. It simulates an online healthcare platform where users can:

- Browse and buy medicines online
- Book diagnostic lab tests and full-body health packages
- Consult certified doctors by chat, audio or video
- Shop COVID essentials and general health care products
- Read doctor-reviewed health articles
- Manage a cart and a simple sign-in flow
- Contact the support team

This is a redesign of the original student project, focused on a consistent design system, richer content, and small UX fixes (missing cart page, broken image paths, inconsistent branding) rather than a rewrite of its purpose.

---

## 🌐 Pages

| Page              | File                   | Description                                                                     |
| ----------------- | ---------------------- | ------------------------------------------------------------------------------- |
| Home              | `index.html`           | Hero search, quick services, categories, best sellers, blog preview, stats, FAQ |
| Medicines         | `medicines.html`       | Filterable medicine catalog with prescription upload                            |
| Health Care       | `healthcare.html`      | Category & condition-based product browsing                                     |
| Lab Tests         | `labtest.html`         | Individual tests + full body health packages                                    |
| Consult Doctors   | `consultdoctor.html`   | Specialities grid, "how it works", top-rated doctors                            |
| Covid Essentials  | `covidessentials.html` | Masks, sanitizers, oximeters, recovery care                                     |
| Blog              | `blog.html`            | Health articles with categories & newsletter signup                             |
| Contact Us        | `contact.html`         | Contact form, quick FAQs, support channels                                      |
| Sign In / Profile | `profile.html`         | OTP-based sign in, social sign-in options                                       |
| Cart              | `addtocart.html`       | Cart review, coupon, order summary (new — was linked but missing)               |

---

## 🎨 Design system

- **Palette** — clinical teal (`#0E7C86`) for trust, warm coral (`#FF6B4A`) for calls to action, deep ink (`#0F2A2E`) for text, warm sand background.
- **Type** — Fraunces (display headings), Inter (body copy), IBM Plex Mono (prices, dosages, stats) — a nod to prescription labels and lab reports.
- **Signature element** — the `.rx-tag`, a dashed-border "prescription label" chip used for discounts, dosage info and category badges across the site.
- All shared styles live in `css/style.css`; shared interactivity (cart badge, toasts, FAQ accordion, back-to-top, demo form submissions) lives in `js/app.js`.

---

## 🛠️ Tech Stack

- HTML5, CSS3 (custom design system, no framework overrides needed beyond Bootstrap's grid)
- Bootstrap 5.3 (layout/grid/utilities only — components are custom-styled)
- Font Awesome 6.5 icons
- Vanilla JavaScript (no build step, no dependencies)

---

## 📁 Folder Structure

```
📦 MedsMitra/
 ┣ 📄 index.html
 ┣ 📄 medicines.html
 ┣ 📄 healthcare.html
 ┣ 📄 labtest.html
 ┣ 📄 consultdoctor.html
 ┣ 📄 covidessentials.html
 ┣ 📄 blog.html
 ┣ 📄 contact.html
 ┣ 📄 profile.html
 ┣ 📄 addtocart.html
 ┣ 📁 css/
 ┃ ┗ 📄 style.css
 ┗ 📁 js/
   ┗ 📄 app.js
```

Product, doctor and blog images are loaded from `picsum.photos` placeholders (seeded, so each stays consistent between reloads) rather than a local `images/` folder — swap these `<img src>` values for real product photography before going to production.

---

## 🚀 How to Run

1. Open the project folder in VS Code (or any editor).
2. Right-click `index.html` → **Open with Live Server**, or just double-click it to open in your browser.
3. No build step, no dependencies to install — it's a static site.

> ⚠️ This is a **static frontend-only demo**. Forms, "Add to cart", checkout and sign-in are simulated in the browser (see `js/app.js`) and don't persist data or call a real backend.

---

## 🔮 Future Scope

- Integrate a real backend (Node.js, Firebase, or Django) and product/order database
- Real authentication (OTP/email/social) with session persistence
- Working cart and checkout with payment gateway integration
- Live chat/video doctor consultations
- Real prescription upload with pharmacist verification workflow

---

## 📌 Author

Originally built by Sneha Vishwakarma. Redesigned with 💻 and a cup of chai.
