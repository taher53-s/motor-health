# Motor Health project notes

- Preserve the existing dashboard and Fetch API when changing deployment infrastructure.
- Vercel uses `api/index.ts` to wrap `worker.mjs`; `vercel.json` routes requests to it.
- Run `npm run build` then `npm test` before deployment. The build requires Python 3 and no extra Python packages.
- Keep `presentation.json` tracked: the original ZIP omitted it, and it was recovered from the supplied worker.
- Emit static files into `public/` because Vercel requires a nonempty output directory.
- Model and benchmark data are bundled read-only. `evidence.sqlite` is an evaluation snapshot, not a writable backend. Prediction requests are not persisted.
- Keep replay recording IDs sorted when generating the worker for deterministic builds.
- Use the user's terminal Git authentication for this repository. GitHub main is connected to Vercel production.
