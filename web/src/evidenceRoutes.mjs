export function evidenceViewerUrl(href) {
  if (!/^\/(?:evidence\/|data\/export-verification\.json)/.test(href || ""))
    return href;
  const [path, hash = ""] = href.split("#");
  if (!/\.(md|json)$/i.test(path)) return href;
  return `/?${new URLSearchParams({ evidence: path })}${hash ? `#${hash}` : ""}`;
}

export function evidenceEntries(manifest) {
  return Object.values(manifest?.evidence || {}).flat();
}

export function resolveEvidenceLink(href, source, manifest, image = false) {
  if (!href) return undefined;
  if (href.startsWith("#")) return href;
  if (/^https?:\/\//i.test(href)) return href;
  if (/^[a-z][a-z\d+.-]*:/i.test(href) || href.startsWith("//"))
    return undefined;
  const resolved = new URL(href, `https://evidence.local/${source}`);
  const path = decodeURIComponent(resolved.pathname.slice(1));
  const mapped = manifest?.evidenceSourceMap?.[path];
  if (mapped) return image ? mapped : evidenceViewerUrl(mapped + resolved.hash);
  if (href.startsWith("/") && /^\/(evidence|media|data)\//.test(href)) {
    return image ? href : evidenceViewerUrl(href);
  }
  const repo = manifest?.experiment?.repository;
  return repo
    ? `${repo}/blob/main/${path.split("/").map(encodeURIComponent).join("/")}${resolved.hash}`
    : undefined;
}

export function headingId(text) {
  return text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s-]/gu, "")
    .trim()
    .replace(/\s+/g, "-");
}

export function completionRows(data) {
  const rows = [];
  for (const [suite, value] of Object.entries(data?.suites || {})) {
    for (const [method, result] of Object.entries(value.results || {})) {
      for (const run of result.runs || [])
        rows.push({ suite, suiteSeed: value.seed, method, ...run });
    }
  }
  return rows;
}

export function trialMethodLabel(method) {
  if (method === "feedback_only") return "Feedback alone";
  const match = method.match(
    /^(source_direct|source_contact|posttrained|original_20k|longer_original|strict_objective)_seed([012])(_finish)?$/,
  );
  if (!match) return method.replace(/_/g, " ");
  const names = {
    source_direct: "Direct transfer",
    source_contact: "Contact adapter",
    posttrained: "Original RL 20k",
    original_20k: "Original RL 20k",
    longer_original: "Longer RL 60k (18 mm goal)",
    strict_objective: "Strict RL 60k (8 mm goal)",
  };
  return `${names[match[1]]} · seed ${match[2]} · ${match[3] ? "with feedback" : "policy alone"}`;
}

export function replaySelection(search) {
  const params = new URLSearchParams(search);
  const match = params
    .get("run")
    ?.match(
      /^(source_direct|source_contact|posttrained|longer_original|strict_objective)_seed([012])(_finish)?$/,
    );
  const task = Number(params.get("task"));
  if (!match || !Number.isInteger(task) || task < 1 || task > 20) return null;
  return {
    strategy:
      match[1] === "source_direct"
        ? "direct"
        : match[1] === "source_contact"
          ? "contact"
          : "postRL",
    trainingVersion:
      match[1] === "longer_original"
        ? "longer"
        : match[1] === "strict_objective"
          ? "strict"
          : "original",
    seed: Number(match[2]),
    trial: task - 1,
    mode: match[3] ? "finish" : "policy",
  };
}

export function trialReplayUrl(row, manifest) {
  if (row.suite !== "paired") return null;
  const run = row.method.replace(/^original_20k/, "posttrained");
  if (!manifest?.runs?.[run]?.trials?.[row.trial]) return null;
  return `/?${new URLSearchParams({ run, task: String(row.trial + 1) })}#lab`;
}
