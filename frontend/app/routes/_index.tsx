import type { MetaFunction } from "@remix-run/node";
import type { ActionFunction } from "@remix-run/node";
import { json } from "@remix-run/node";
import { useActionData, Form } from "@remix-run/react";
import StockChart from "../components/charts/StockChart";

export const meta: MetaFunction = () => {
	return [
		{ title: "New Remix App" },
		{ name: "description", content: "Welcome to Remix!" },
	];
};

export const action: ActionFunction = async ({ request }) => {
	const form = await request.formData();
	const ticker = form.get("ticker")?.toString();
	const timeframe = form.get("timeframe")?.toString();

	// Node fetch용 절대 URL 생성
	const urlObj = new URL(request.url);
	const apiUrl = new URL("/api/stock/chart-data", urlObj).href;

	const response = await fetch(apiUrl, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({ ticker, timeframe }),
	});
	if (!response.ok) throw new Response("Failed to fetch chart data", { status: response.status });
	const data = await response.json();
	return json(data);
};

export default function Index() {
	const actionData = useActionData<typeof action>();

	return (
		<div className="flex h-screen items-center justify-center">
			<div className="flex flex-col items-stretch gap-16 w-full max-w-5xl mx-auto">
				<header className="flex flex-col items-center gap-9">
					<h1 className="leading text-2xl font-bold text-gray-800 dark:text-gray-100">
						 Test
					</h1>
				</header>
				<Form method="post" className="mb-4 flex gap-2">
					<input name="ticker" type="text" defaultValue="TSLY" placeholder="Ticker" className="border px-2 py-1 rounded" />
					<input name="timeframe" type="text" defaultValue="1mo" placeholder="Timeframe" className="border px-2 py-1 rounded" />
					<button type="submit" className="bg-blue-500 text-white px-4 py-1 rounded">Load</button>
				</Form>
				<div className="flex flex-col items-stretch">
					{actionData && <StockChart data={actionData} />}
				</div>
			</div>
		</div>
	);
}