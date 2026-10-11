import React, { lazy, Suspense } from "react";
import { createRoot } from "react-dom/client";
import App from "./App.jsx";
import "./style.css";
const EvidenceReader = lazy(() => import("./EvidenceReader.jsx"));
const evidence = new URLSearchParams(location.search).get("evidence");
createRoot(document.getElementById("root")).render(
  evidence ? (
    <Suspense
      fallback={<p className="ev-loading">Opening the evidence viewer…</p>}
    >
      <EvidenceReader path={evidence} />
    </Suspense>
  ) : (
    <App />
  ),
);
