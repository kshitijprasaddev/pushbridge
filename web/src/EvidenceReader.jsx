import React, { useEffect, useMemo, useState } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  completionRows,
  evidenceEntries,
  evidenceViewerUrl,
  headingId,
  resolveEvidenceLink,
  trialMethodLabel,
  trialReplayUrl,
} from "./evidenceRoutes.mjs";
import "./evidence.css";

function nodeText(node) {
  return typeof node === "string"
    ? node
    : Array.isArray(node)
      ? node.map(nodeText).join("")
      : node?.props
        ? nodeText(node.props.children)
        : "";
}
function JsonBranch({ name, value, depth = 0 }) {
  const [open, setOpen] = useState(depth < 1);
  if (value === null || typeof value !== "object") {
    return (
      <div className="ev-json-value">
        <span>{name}</span>
        <code>{String(value)}</code>
      </div>
    );
  }
  const entries = Object.entries(value);
  return (
    <details
      className="ev-json-branch"
      open={open}
      onToggle={(e) => setOpen(e.currentTarget.open)}
    >
      <summary>
        {name}
        <small>
          {entries.length} {Array.isArray(value) ? "items" : "fields"}
        </small>
      </summary>
      {open && (
        <div>
          {entries.map(([key, child]) => (
            <JsonBranch key={key} name={key} value={child} depth={depth + 1} />
          ))}
        </div>
      )}
    </details>
  );
}
function CompletionTable({ data, manifest }) {
  const rows = useMemo(() => completionRows(data), [data]);
  const [suite, setSuite] = useState("all"),
    [method, setMethod] = useState("all"),
    [outcome, setOutcome] = useState("all");
  const [page, setPage] = useState(0);
  const filtered = rows.filter(
    (r) =>
      (suite === "all" || r.suite === suite) &&
      (method === "all" || r.method === method) &&
      (outcome === "all" || r.success === (outcome === "success")),
  );
  const shown = filtered.slice(page * 30, (page + 1) * 30);
  const update = (fn, value) => {
    fn(value);
    setPage(0);
  };
  if (!rows.length) return null;
  return (
    <section className="ev-completions">
      <h2>Inspect individual trials</h2>
      <p>
        Each row is a recorded physics rollout. The same task conditions repeat
        across training seeds. A successful trial must pass the containment and
        hold checks.
      </p>
      <div className="ev-stat-row">
        <div>
          <strong>{rows.length}</strong>
          <span>saved trials</span>
        </div>
        <div>
          <strong>{rows.filter((r) => r.success).length}</strong>
          <span>completed</span>
        </div>
        <div>
          <strong>{rows.filter((r) => !r.success).length}</strong>
          <span>not completed</span>
        </div>
      </div>
      <div className="ev-filters">
        <label>
          Task suite
          <select
            value={suite}
            onChange={(e) => update(setSuite, e.target.value)}
          >
            <option value="all">All suites</option>
            {[...new Set(rows.map((r) => r.suite))].map((s) => (
              <option key={s}>{s}</option>
            ))}
          </select>
        </label>
        <label>
          Policy and controller
          <select
            value={method}
            onChange={(e) => update(setMethod, e.target.value)}
          >
            <option value="all">All methods</option>
            {[...new Set(rows.map((r) => r.method))].map((s) => (
              <option key={s} value={s}>
                {trialMethodLabel(s)}
              </option>
            ))}
          </select>
        </label>
        <label>
          Outcome
          <select
            value={outcome}
            onChange={(e) => update(setOutcome, e.target.value)}
          >
            <option value="all">All outcomes</option>
            <option value="success">Completed</option>
            <option value="failure">Not completed</option>
          </select>
        </label>
      </div>
      <div className="ev-table-scroll">
        <table>
          <thead>
            <tr>
              <th>Suite</th>
              <th>Method</th>
              <th>Task</th>
              <th>Outcome</th>
              <th>Final error</th>
              <th>Hold ticks</th>
              <th>Recorded replay</th>
            </tr>
          </thead>
          <tbody>
            {shown.map((r, i) => (
              <tr key={`${r.suite}-${r.method}-${r.trial}-${i}`}>
                <td>{r.suite}</td>
                <td>{trialMethodLabel(r.method)}</td>
                <td>{String(r.trial + 1).padStart(2, "0")}</td>
                <td className={r.success ? "ev-pass" : "ev-fail"}>
                  {r.success ? "Complete" : "Incomplete"}
                </td>
                <td>{r.final_distance_mm.toFixed(2)} mm</td>
                <td>{r.verification_hold_ticks ?? "n/a"}</td>
                <td>
                  {trialReplayUrl(r, manifest) ? (
                    <a href={trialReplayUrl(r, manifest)}>View in 3D ↗</a>
                  ) : (
                    <span className="ev-table-note">Source data only</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="ev-pagination">
        <span>
          {filtered.length} matching trials · page {page + 1} of{" "}
          {Math.max(1, Math.ceil(filtered.length / 30))}
        </span>
        <button disabled={!page} onClick={() => setPage((p) => p - 1)}>
          Previous
        </button>
        <button
          disabled={(page + 1) * 30 >= filtered.length}
          onClick={() => setPage((p) => p + 1)}
        >
          Next
        </button>
      </div>
    </section>
  );
}
export default function EvidenceReader({ path }) {
  const [manifest, setManifest] = useState(null),
    [entry, setEntry] = useState(null),
    [text, setText] = useState(""),
    [error, setError] = useState("");
  useEffect(() => {
    const controller = new AbortController();
    async function load() {
      const res = await fetch("/data/manifest.json", {
        signal: controller.signal,
        cache: "no-cache",
      });
      if (!res.ok) throw new Error("The evidence index could not be loaded.");
      const index = await res.json();
      const canonical =
        index.evidenceSourceMap?.[path.replace(/^\/evidence\//, "")] || path;
      const file = evidenceEntries(index).find(
        (e) => e.url === canonical && /\.(md|json)$/.test(e.url),
      );
      if (!file)
        throw new Error(
          "This document is not in the published evidence library.",
        );
      setManifest(index);
      setEntry(file);
      const content = await fetch(file.url, {
        signal: controller.signal,
        cache: "no-cache",
      });
      if (!content.ok)
        throw new Error(
          `The source document could not be loaded (${content.status}).`,
        );
      setText(await content.text());
      document.title = `${file.label} | PushBridge evidence`;
    }
    load().catch((e) => {
      if (e.name !== "AbortError") setError(e.message);
    });
    return () => controller.abort();
  }, [path]);
  useEffect(() => {
    if (text && location.hash)
      requestAnimationFrame(() =>
        document
          .getElementById(decodeURIComponent(location.hash.slice(1)))
          ?.scrollIntoView(),
      );
  }, [text]);
  const json = useMemo(() => {
    try {
      return path.endsWith(".json") && text ? JSON.parse(text) : null;
    } catch {
      return null;
    }
  }, [text, path]);
  const toc = useMemo(
    () =>
      text
        .split("\n")
        .filter((l) => /^##\s/.test(l))
        .map((l) => {
          const title = l.replace(/^##\s+/, "");
          return { title, id: headingId(title) };
        }),
    [text],
  );
  const links = evidenceEntries(manifest).filter((e) =>
    /\.(md|json)$/.test(e.url),
  );
  const components = {
    a: ({ href, children }) => {
      const resolved = resolveEvidenceLink(href, entry?.source, manifest);
      return (
        <a
          href={resolved}
          {...(/^https?:/.test(resolved || "")
            ? { target: "_blank", rel: "noreferrer" }
            : {})}
        >
          {children}
        </a>
      );
    },
    img: ({ src, alt }) => (
      <img
        src={resolveEvidenceLink(src, entry?.source, manifest, true)}
        alt={alt || ""}
        loading="lazy"
      />
    ),
    table: ({ children }) => (
      <div className="ev-table-scroll">
        <table>{children}</table>
      </div>
    ),
    ...Object.fromEntries(
      [1, 2, 3, 4].map((level) => [
        `h${level}`,
        ({ children }) =>
          React.createElement(
            `h${level}`,
            { id: headingId(nodeText(children)) },
            children,
          ),
      ]),
    ),
  };
  return (
    <div className="ev-page">
      <header className="ev-header">
        <a className="ev-brand" href="/">
          PushBridge<span>EVIDENCE LIBRARY</span>
        </a>
        <nav>
          <a href="/#mujoco">MuJoCo recordings</a>
          <a href="/#lab">Back to the lab ↗</a>
        </nav>
      </header>
      <div className="ev-layout">
        <aside className="ev-sidebar">
          <span className="eyebrow">DOCUMENTS & RESULTS</span>
          <nav aria-label="Evidence library">
            {links.map((file) => (
              <a
                key={file.url}
                href={evidenceViewerUrl(file.url)}
                aria-current={file.url === entry?.url ? "page" : undefined}
              >
                {file.label}
                <small>
                  {file.url.endsWith(".md") ? "Article" : "Measured data"}
                </small>
              </a>
            ))}
          </nav>
          {toc.length > 0 && (
            <div className="ev-toc">
              <span className="eyebrow">ON THIS PAGE</span>
              {toc.map(({ title, id }) => (
                <a key={id} href={`#${id}`}>
                  {title}
                </a>
              ))}
            </div>
          )}
        </aside>
        <main className="ev-document">
          <div className="ev-document-bar">
            <span>
              {path.endsWith(".md") ? "RESEARCH ARTICLE" : "EVIDENCE EXPLORER"}
            </span>
            {entry && (
              <a href={entry.url} download>
                Download source {path.endsWith(".md") ? ".md" : ".json"} ↓
              </a>
            )}
          </div>
          {error ? (
            <div role="alert" className="ev-error">
              <h1>Document unavailable</h1>
              <p>{error}</p>
              <a href="/#reproducibility">Return to the evidence library</a>
            </div>
          ) : !text ? (
            <p className="ev-loading" role="status">
              Loading the source document…
            </p>
          ) : json ? (
            <>
              <h1>{entry.label}</h1>
              {json.status && (
                <div className="ev-audit-status">
                  Recorded audit status: <strong>{json.status}</strong>
                </div>
              )}
              {json.scope && <p>{json.scope}</p>}
              {Array.isArray(json.checks) && (
                <ul className="ev-audit-checks">
                  {json.checks.map((check, i) => (
                    <li key={i}>
                      {typeof check === "string"
                        ? check
                        : JSON.stringify(check)}
                    </li>
                  ))}
                </ul>
              )}
              <CompletionTable data={json} manifest={manifest} />
              <h2>Explore the source fields</h2>
              <p>
                Expand a section to inspect its protocol and recorded values.
                The original JSON download is available above.
              </p>
              <JsonBranch name="Source data" value={json} />
            </>
          ) : (
            <Markdown remarkPlugins={[remarkGfm]} components={components}>
              {text}
            </Markdown>
          )}
          {entry?.sha256 && (
            <footer className="ev-source-note">
              Source:{" "}
              <a
                href={`${manifest.experiment.repository}/blob/main/${entry.source}`}
                target="_blank"
                rel="noreferrer"
              >
                {entry.source}
              </a>
              <details>
                <summary>File verification</summary>
                <code>SHA-256 {entry.sha256}</code>
              </details>
            </footer>
          )}
        </main>
      </div>
    </div>
  );
}
