import { vitePlugin as remix } from "@remix-run/dev";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vite";
import tsconfigPaths from "vite-tsconfig-paths";
import dotenv from "dotenv";

dotenv.config();


export default defineConfig({
	define: {
		'process.env.API_BASE_URL': JSON.stringify(process.env.API_BASE_URL),
	},
	plugins: [
		remix({
			future: {
				v3_fetcherPersist: true,
				v3_relativeSplatPath: true,
				v3_throwAbortReason: true,
				v3_singleFetch: true,
				v3_lazyRouteDiscovery: true,
			},
		}),
		tsconfigPaths(),
		tailwindcss(),
		{
			name: 'handle-well-known',
			configureServer(server) {
				server.middlewares.use('/.well-known/appspecific/com.chrome.devtools.json', (req, res, next) => {
					res.statusCode = 204;
					res.end();
				});
			},
		},
	],
	server: {
		proxy: {
			"/api": {
				target: process.env.API_BASE_URL,
				changeOrigin: true,
				secure: false,
				rewrite: (path) => path.replace(/^\/api/, ""),
			},
		},
	},	
	resolve: {
		dedupe: ["react", "react-dom"], // ✅ React 중복 제거
	},
	optimizeDeps: {
		entries: ["src/**/*.tsx", "src/**/*.ts"],
	},
});
