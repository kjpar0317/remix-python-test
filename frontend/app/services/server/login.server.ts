import type { ActionFunctionArgs } from "@remix-run/node";

import { ssrFetcher } from "~/lib/utils";

export async function action({
	request,
}: ActionFunctionArgs): Promise<Response> {
	const formData = await request.formData();
	const { accessToken } = await ssrFetcher(request, "/api/auth/login", "POST", formData);
	// const isProduction = process.env.NODE_ENV === "production";
	// const isHttps = process.env.API_BASE_URL?.startsWith("https://");
	

	return new Response(null, {
		status: 302,
		headers: {
			"Set-Cookie": `token=${accessToken}; Path=/; HttpOnly; SameSite=Lax; Secure;`,
			Location: "/",	// ssr은 쿠키 바로 전달 안됨
		},
	});
	// return new Response(null, {
	// 	status: 302,
	// 	headers: {
	// 		"Set-Cookie": `token=${accessToken}; Path=/; HttpOnly; SameSite=None; ${isProduction && isHttps ? "Secure;" : ""}`,
	// 		Location: "/dashboard",
	// 	},
	// });

}
