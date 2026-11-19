#!/usr/bin/env node
/**
 * Frontend script to generate TypeScript definitions from FastAPI OpenAPI schema.
 * Can be invoked via `npm run generate-types` or `npm run generate:types`.
 */

const { spawnSync } = require("child_process");
const fs = require("fs");
const path = require("path");

const rootDir = path.resolve(__dirname, "..", "..");
const backendVenvPython = path.join(rootDir, "backend", ".venv", "bin", "python");
const generateScript = path.join(rootDir, "backend", "scripts", "generate_types.py");

// Determine suitable Python interpreter
let pythonExecutable = "python3";
if (fs.existsSync(backendVenvPython)) {
  pythonExecutable = backendVenvPython;
} else if (process.env.PYTHON) {
  pythonExecutable = process.env.PYTHON;
}

const args = [generateScript, ...process.argv.slice(2)];
const result = spawnSync(pythonExecutable, args, {
  cwd: path.join(rootDir, "backend"),
  stdio: "inherit",
  env: {
    ...process.env,
    PYTHONPATH: path.join(rootDir, "backend"),
    STATIC_DIR: path.join(rootDir, "backend", "static"),
  },
});

if (result.status !== 0) {
  process.exit(result.status || 1);
}
