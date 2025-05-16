import {
	Links,
	Meta,
	Outlet,
	Scripts,
	ScrollRestoration,
	useMatches,
} from "@remix-run/react";
import type { LinksFunction } from "@remix-run/node";

import styles from "~/tailwind.css?url"
import TemplateLayout from "./components/templates/TemplateLayout";

export const links: LinksFunction = () => [
	{ rel: "preconnect", href: "https://fonts.googleapis.com" },
	{
		rel: "preconnect",
		href: "https://fonts.gstatic.com",
		crossOrigin: "anonymous",
	},
	{
		rel: "stylesheet",
		href: "https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,100..900;1,14..32,100..900&display=swap",
	},
	{ rel: "stylesheet", href: styles },
];

export default function App() {
	const matches = useMatches();
  	// 현재 라우트 중 noLayout 설정이 되어 있는지 확인
  	// biome-ignore lint/suspicious/noExplicitAny: <explanation>
  	const noLayout = matches.some((match: any) => match.handle?.noLayout);
	const content = <Outlet />;

	return (
	  <html lang="ko">
		<head>
			<meta charSet="utf-8" />
			<meta name="viewport" content="width=device-width, initial-scale=1" />
			<Meta />
			<Links />
		</head>
		<body className="w-full h-full m-0">
			{noLayout ? content : <TemplateLayout>{content}</TemplateLayout>}
			<ScrollRestoration />
			<Scripts />
		</body>
	  </html>
	);
}
