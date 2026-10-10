# Third-party components

The frontend uses React and Three.js (MIT), Vite (MIT), Inter (SIL Open Font License), and IBM Plex Mono (SIL Open Font License). Exact versions and dependency integrity are recorded in `pnpm-lock.yaml`.

Font license texts are included under `public/licenses/` and copied into the production build. Robot geometry, lab materials and labels are created in source code. Fonts, replay data and videos are served from the same deployment; the rendered app requests no remote models, environment maps or fonts.

Node 24.x and pnpm 11.25.0 are build tools. They are not required on a reviewer's device. The Vercel configuration invokes that exact pnpm version for both dependency installation and the build, with a frozen lockfile.
