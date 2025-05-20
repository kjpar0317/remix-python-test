import type { ActionFunctionArgs } from "@remix-run/node";

export async function action({
	request,
}: ActionFunctionArgs): Promise<Response> {
	const formData = await request.formData();

	// Node fetch용 절대 URL 생성
	const urlObj = new URL(request.url);
	const apiUrl = `${urlObj.origin}/api/auth/login`;
	const res = await fetch(apiUrl, {
		method: "POST",
		// headers: { "Content-Type": "application/json" },
		body: formData,
	});

	if (!res.ok) {
		throw new Response("Unauthorized", { status: 401 });
	}

	const { accessToken } = await res.json();

	return new Response(null, {
		status: 302,
		headers: {
			"Set-Cookie": `token=${accessToken}; Path=/; HttpOnly; Secure; SameSite=Lax`,
			Location: "/dashboard",
		},
	});
}
