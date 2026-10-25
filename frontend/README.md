# MedTrust Frontend

Frontend for **MedTrust — Trust-Aware Explainable AI for Chest X-Ray Disease Detection**: a React + TypeScript
research interface for multi-label chest X-ray analysis, uncertainty, image quality, out-of-distribution detection,
calibration and explainability.

This project is **frontend only**. It contains no model and no backend. It talks to an existing or future backend over
a configurable REST API, and it runs fully in **demo mode** (clearly labelled sample data) with no backend at all.

> Research use only. Not a medical device, not clinically validated, not for diagnosis.

---

## 1. Quick start

**Requirements:** Node.js **18.18 or newer** (20 LTS or 22 recommended) and npm. Check with `node -v`.

### Windows PowerShell

```powershell
cd medtrust-frontend
npm install
Copy-Item .env.example .env
npm.cmd run dev
```

### macOS / Linux

```bash
cd medtrust-frontend
npm install
cp .env.example .env
npm run dev
```

Open the address Vite prints (normally <http://localhost:5173>). The default `.env.example` enables demo mode, so
everything works without a backend.

If `.env` already exists, do not overwrite it. Open it and edit the two values you need:

| Variable | Meaning |
|---|---|
| `VITE_API_BASE_URL` | Backend address, e.g. `http://127.0.0.1:8000` (no trailing slash) |
| `VITE_DEMO_MODE` | `true` = sample data, no backend. `false` = use the backend |

Restart `npm run dev` after changing `.env`; Vite reads it at start-up.

> PowerShell tip: if `npm` is blocked by the execution policy, use `npm.cmd` as shown above.

## 2. Commands

| Task | PowerShell | macOS / Linux |
|---|---|---|
| Dev server | `npm.cmd run dev` | `npm run dev` |
| Type-check | `npm.cmd run typecheck` | `npm run typecheck` |
| Lint | `npm.cmd run lint` | `npm run lint` |
| Tests (once) | `npm.cmd test` | `npm test` |
| Tests (watch) | `npm.cmd run test:watch` | `npm run test:watch` |
| Production build | `npm.cmd run build` | `npm run build` |
| Preview the build | `npm.cmd run preview` | `npm run preview` |
| Everything above | `npm.cmd run check` | `npm run check` |

`build` type-checks first, then writes static files to `dist/`. Environment variables are baked in at build time, so
rebuild after changing `.env`.

## 3. Configuration

All settings are `VITE_*` variables (see `.env.example`). Never put secrets in them: they are visible in the browser.

| Variable | Default | Purpose |
|---|---|---|
| `VITE_API_BASE_URL` | `http://127.0.0.1:8000` | Backend base URL |
| `VITE_API_PREFIX` | `/api/v1` | Path prefix for all endpoints |
| `VITE_DEMO_MODE` | `false` in code, `true` in `.env.example` | Sample-data mode |
| `VITE_REQUEST_TIMEOUT_MS` | `15000` | Timeout for ordinary requests |
| `VITE_ANALYSIS_TIMEOUT_MS` | `120000` | Timeout for upload + analysis |
| `VITE_MAX_UPLOAD_MB` | `15` | Largest accepted image |
| `VITE_ENABLE_REMEMBER_ME` | `false` | Show "Remember me" (only if your backend honours `remember_me`) |
| `VITE_INSTITUTION`, `VITE_CONTACT_EMAIL`, `VITE_PROJECT_TEAM` | empty | About page and footer details. `VITE_PROJECT_TEAM` is JSON, e.g. `[{"name":"Full Name","role":"Role"}]` |

## 4. Demo mode

With `VITE_DEMO_MODE=true`:

- A banner on every page states that data is sample data and sign-in is simulated.
- Analyses, history and reports are generated locally, deterministically from the uploaded file name and size. **None
  of it comes from a model.** Every such result is labelled "Sample data".
- Your image never leaves the browser.
- Sign-in accepts any well-formed email. It is **not secure authentication**. Passwords are never stored; only a
  display profile is kept in `sessionStorage` and disappears when the tab closes.

With `VITE_DEMO_MODE=false` the demo code is never called. A failed request shows a real error and **never** falls
back to sample results. The demo code lives in `src/demo/` and is selected in exactly one place,
`src/services/index.ts`; the real API modules do not import it.

## 5. Connecting your backend

1. Start the backend and note its address.
2. Set `VITE_API_BASE_URL` to it and `VITE_DEMO_MODE=false` in `.env`, then restart the dev server.
3. **CORS and cookies.** The client sends `credentials: "include"`, so the backend must:
   - allow the exact frontend origin (e.g. `http://localhost:5173`), not `*`;
   - send `Access-Control-Allow-Credentials: true`;
   - set the session as a server-managed **HTTP-only** cookie (use `SameSite=None; Secure` if the frontend and API
     are on different sites over HTTPS).

   The frontend never stores passwords or tokens in browser storage.

### Endpoints the frontend calls

These are integration contracts, not a claim that the backend already implements them. Paths are defined in
`src/services/endpoints.ts`.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/v1/health/live` | Liveness |
| GET | `/api/v1/health/ready` | Readiness (503 = alive but not ready) |
| POST | `/api/v1/auth/register` | Create account (`full_name`, `email`, `password`) |
| POST | `/api/v1/auth/login` | Sign in (`email`, `password`); sets the session cookie |
| POST | `/api/v1/auth/logout` | Sign out |
| GET | `/api/v1/auth/me` | Current user; 401 when signed out |
| POST | `/api/v1/analyses` | Multipart upload, field name `file` |
| GET | `/api/v1/analyses` | List the current user's analyses (`page`, `page_size`, `q`, `sort_by`, `order`) |
| GET | `/api/v1/analyses/:id` | Analysis details |
| GET | `/api/v1/analyses/:id/report` | Report |
| GET | `/api/v1/analyses/:id/artifacts/:name` | Image artifacts (`original`, Grad-CAM, …) |

### Adapting to your response shape

The backend schema was not supplied, so responses are parsed defensively. **All parsing is in
`src/services/normalizers.ts`**; the accepted shape is documented in the comment at the top of that file. It accepts
snake_case or camelCase, predictions as an array, an object or parallel maps, and ignores probabilities outside 0–1.
Anything not returned is shown as "not evaluated" or "not provided"; nothing is invented. To adapt, edit that file
(and `endpoints.ts`, `UPLOAD_FIELD_NAME`).

If `POST /analyses` returns only an id and a `queued`/`processing` status, the client polls
`GET /analyses/:id` until it finishes (up to `VITE_ANALYSIS_TIMEOUT_MS`).

### Security notes

- Route protection in the browser (`ProtectedRoute`) is a convenience only. The backend **must** enforce
  authorization on every request, in particular on `/analyses/:id`, `/report` and `/artifacts`. A `403` and a `404`
  are shown identically so report IDs cannot be probed.
- A `401` on an authenticated request signs the user out and shows a "session expired" message on the login page.

## 6. Project structure

```text
src/
  app/          App, router, providers
  components/
    layout/     Navbar, Footer, Layout, ProtectedRoute, AuthLayout
    ui/         Button, TextField, Dialog, Alert, Badge, ...
    upload/     ImageDropzone, ImagePreview
    analysis/   DiseaseProbabilityList, QualityAssessment, UncertaintyPanel,
                OODPanel, CalibrationPanel, ExplanationViewer, ...
    feedback/   Empty/Error states, demo banner and badge, API status
  pages/        Home, Login, Signup, Analyze, Report, History, About, Profile, ErrorPages
  services/     apiClient, authApi, analysisApi, endpoints, normalizers, errors
  demo/         Isolated sample data and demo implementations of the services
  context/      AuthContext
  hooks/        useAuth, useAnalysisRun, useAsyncData, ...
  types/        auth, analysis, api
  lib/          validation, formatting, metric explanations
  styles/       design-tokens.css, globals.css
  test/         Vitest + React Testing Library suites
```

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `npm` is blocked in PowerShell | Use `npm.cmd`, or run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `Unsupported engine` / syntax errors | Upgrade Node to 18.18+ (`node -v`) |
| Port 5173 is busy | `npm run dev -- --port 5180` |
| Changed `.env` but nothing changed | Restart the dev server (rebuild for production) |
| Banner says "Demo mode" but you want the backend | Set `VITE_DEMO_MODE=false` and restart |
| "The MedTrust service could not be reached" | Backend not running, wrong `VITE_API_BASE_URL`, or CORS not allowing this origin |
| Login succeeds but you are logged out on refresh | Cookie not being kept: check `SameSite`, `Secure`, `Access-Control-Allow-Credentials` |
| Images show "Image unavailable" in a report | The artifact endpoint returned an error; check `/analyses/:id/artifacts/original` |
| A result field is "not evaluated" | The backend did not return it; check the shape against `normalizers.ts` |
| Clean reinstall | Delete `node_modules` and run `npm install` again |

## 8. Known limitations

- Needs a backend implementing the contract above for real results; none is included.
- Account deletion, profile editing, password change and password recovery are not implemented. The interface says so
  rather than offering controls that do nothing.
- History supports search, sort and pagination but not deletion, because no delete endpoint was specified.
- Analysis stage markers (validation, prediction, reliability, report) are visual progress only; the upload
  percentage is real.
- "Print or save as PDF" uses the browser's print dialog; "Download data (JSON)" exports the parsed report.
- Light theme only. The monochrome identity is the primary design; no dark-mode switch is provided.
- Tests run in jsdom; layout was checked in Chromium at 1440 px and 390 px widths but there is no automated
  visual-regression suite.
