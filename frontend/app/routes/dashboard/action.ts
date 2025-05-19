import type { ActionFunction } from "@remix-run/node";
import { getToken } from "~/services/auth/auth.server";

export const action: ActionFunction = async ({ request }) => {
  const form = await request.formData();
  const ticker = form.get("ticker")?.toString();
  const timeframeValue = form.get("timeframeValue")?.toString() || "1";
  const timeframeUnit = form.get("timeframeUnit")?.toString() || "mo";
  const timeframe = `${timeframeValue}${timeframeUnit}`;

  // Node fetch용 절대 URL 생성
  const urlObj = new URL(request.url);
  // const apiUrl = new URL("/api/stock/chart-data", urlObj).href;
  const apiUrl = `${urlObj.origin}/api/stock/chart-data`;
  const token = getToken(request);

  const response = await fetch(apiUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json", "Authorization": `Bearer ${token}` },
    body: JSON.stringify({ ticker, timeframe }),
    // credentials: 'include'  // 쿠키를 포함하여 요청
  });
  if (!response.ok)
    throw new Response("Failed to fetch chart data", {
      status: response.status,
    });
  const data = await response.json();

  return { ticker, timeframe, ...data };
};