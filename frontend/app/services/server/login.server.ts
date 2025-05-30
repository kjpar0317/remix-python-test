import type { ActionFunctionArgs } from "@remix-run/node";

import { ssrFetcher } from "~/lib/utils";

export async function action({
	request,
}: ActionFunctionArgs): Promise<Response> {
	const formData = await request.formData();

	try {
		const response = await ssrFetcher(request, "/api/auth/login", "POST", formData);

		if(response?.access_token) {
			return new Response(null, {
				status: 302,
				headers: {
					"Set-Cookie": `token=${response.access_token}; Path=/; HttpOnly; SameSite=Lax; Secure;`,
					Location: "/",	// ssr은 쿠키 바로 전달 안됨
				},
			});
		}

		return new Response(null, {
			status: 302,
			headers: {
				Location: `/login?error=${response}`
			}
		});
	} catch (error: any) {
		console.log(error)
		const errorDetail = error && 'Unknown error';
		console.log('Error:', errorDetail);
		return new Response(null, {
			status: 302,
			headers: {
				Location: `/login?error=${encodeURIComponent(errorDetail)}`
			}
		});
	}
}
