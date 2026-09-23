import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const packageRoot = path.join(repoRoot, "src", "movie_review_factory");
const templatePath = path.join(repoRoot, "scripts", "onefile-launcher-template.js");
const outputPath = path.join(repoRoot, "movie-review-factory.js");

const runtimeFiles = [
  "__init__.py",
  "cli.py",
  "content_agent.py",
  "localization.py",
  "models.py",
  "pipeline.py",
  "webapp.py",
];

const bundle = {};
for (const name of runtimeFiles) {
  const bytes = fs.readFileSync(path.join(packageRoot, name));
  bundle["movie_review_factory/" + name] = bytes.toString("base64");
}
bundle["pyproject.toml"] = fs.readFileSync(path.join(repoRoot, "pyproject.toml")).toString("base64");

const pyproject = fs.readFileSync(path.join(repoRoot, "pyproject.toml"), "utf8");
const versionMatch = pyproject.match(/^version\s*=\s*"([^"]+)"/m);
const version = versionMatch ? versionMatch[1] : "0.0.0";
const bundleJson = JSON.stringify(bundle);
const hash = crypto.createHash("sha256").update(bundleJson).digest("hex");

let launcher = fs.readFileSync(templatePath, "utf8");
launcher = launcher
  .replace("__MRF_VERSION__", JSON.stringify(version))
  .replace("__MRF_HASH__", JSON.stringify(hash))
  .replace("__MRF_BUNDLE__", bundleJson);

if (/__MRF_(VERSION|HASH|BUNDLE)__/.test(launcher)) {
  throw new Error("one-file launcher template contains unresolved placeholders");
}

const asciiLauncher = launcher.replace(
  /[^\x00-\x7F]/g,
  (char) => "\\u" + char.charCodeAt(0).toString(16).padStart(4, "0"),
);
fs.writeFileSync(outputPath, asciiLauncher, "ascii");
const info = {
  output: outputPath,
  bytes: fs.statSync(outputPath).size,
  version,
  hash,
  embeddedFiles: Object.keys(bundle),
};
console.log(JSON.stringify(info, null, 2));
