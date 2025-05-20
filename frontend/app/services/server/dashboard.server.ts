import type { ActionFunction } from "@remix-run/node";

export const action: ActionFunction = async ({ request }) => {
	const form = await request.formData();
	const ticker = form.get("ticker")?.toString();
	const timeframeValue = form.get("timeframeValue")?.toString() || "1";
	const timeframeUnit = form.get("timeframeUnit")?.toString() || "mo";
	const timeframe = `${timeframeValue}${timeframeUnit}`;
  const cookie = request.headers.get('cookie') || '';

	// Node fetch용 절대 URL 생성
	const urlObj = new URL(request.url);
	// const apiUrl = new URL("/api/stock/chart-data", urlObj).href;
	const apiUrl = `${urlObj.origin}/api/stock/chart-data`;

	const response = await fetch(apiUrl, {
		method: "POST",
		headers: {
			"Content-Type": "application/json",
			Cookie: cookie,
		},
		body: JSON.stringify({ ticker, timeframe }),
	});
	if (!response.ok)
		throw new Response("Failed to fetch chart data", {
			status: response.status,
		});
	const data = await response.json();

	return { ticker, timeframe, ...data };
};
