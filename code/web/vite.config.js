import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';

function buildFingerprint() {
  return {
    name: 'cracklab-build-fingerprint',
    closeBundle() {
      const base=process.cwd();
      const walk=directory=>fs.readdirSync(directory,{withFileTypes:true}).flatMap(entry=>entry.isDirectory()?walk(path.join(directory,entry.name)):[path.join(directory,entry.name)]);
      const files=['index.html','package.json','package-lock.json','vite.config.js',...walk('src'),...walk('public')];
      const hashes=Object.fromEntries(files.sort().map(file=>[file.split(path.sep).join('/'),createHash('sha256').update(fs.readFileSync(path.join(base,file))).digest('hex')]));
      fs.writeFileSync(path.join(base,'dist/build_inputs.json'),JSON.stringify({schema_version:1,files:hashes},null,2));
    },
  };
}

export default defineConfig({
  plugins: [react(), buildFingerprint()],
  base: '/static/',
  server: { proxy: { '/api': 'http://127.0.0.1:8765' } },
  build: { outDir: 'dist', emptyOutDir: true },
});
