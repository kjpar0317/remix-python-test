import type { ActionFunctionArgs } from "@remix-run/node";

import { ssrFetcher } from "~/lib/utils";

export async function action({
	request,
}: ActionFunctionArgs): Promise<Response> {
	const formData = await request.formData();
	const { access_token } = await ssrFetcher(request, "/api/auth/login", "POST", formData);

	return new Response(null, {
		status: 302,
		headers: {
			"Set-Cookie": `token=${access_token}; Path=/; HttpOnly; SameSite=Lax; Secure;`,
			Location: "/",	// ssr은 쿠키 바로 전달 안됨
		},
	});
}
