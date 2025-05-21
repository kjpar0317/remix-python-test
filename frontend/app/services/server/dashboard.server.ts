import type { ActionFunction, ActionFunctionArgs } from "@remix-run/node";

import { ssrFetcher } from "~/lib/utils";

export const action: ActionFunction = async ({ request }: ActionFunctionArgs) => {
	const form = await request.formData();
	const ticker = form.get("ticker")?.toString();
	const timeframeValue = form.get("timeframeValue")?.toString() || "1";
	const timeframeUnit = form.get("timeframeUnit")?.toString() || "mo";
	const timeframe = `${timeframeValue}${timeframeUnit}`;

	console.log(request.headers);
	console.log(request.headers.get("cookie"));
	const data = await ssrFetcher(request, "/api/stock/chart-data", "POST", { ticker, timeframe });

	return { ticker, timeframe, ...data };
};
