// Embeddable widget -> dist-widget/widget.js (one self-contained file, React bundled in).
// The FastAPI server serves it at /widget/widget.js.
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  define: { "process.env.NODE_ENV": JSON.stringify("production") },
  publicDir: false,
  build: {
    outDir: "dist-widget",
    emptyOutDir: true,
    lib: {
      entry: "src/widget.tsx",
      name: "RagChatWidget",
      formats: ["iife"],
      fileName: () => "widget.js",
    },
  },
});
