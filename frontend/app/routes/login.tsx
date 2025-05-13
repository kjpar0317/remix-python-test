import { useFetcher } from "@remix-run/react";

export default function Login() {
	const fetcher = useFetcher();

	return (
		<fetcher.Form method="post">
			<input name="username" />
			<input name="password" type="password" />
			<button type="submit">Login</button>
		</fetcher.Form>
	);
}

export async function action({ request }: { request: Request }) {
	const formData = await request.formData();
	const username = formData.get("username");
	const password = formData.get("password");

	const res = await fetch("http://localhost:8000/api/login", {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({ username, password }),
	});

	if (!res.ok) {
		throw new Response("Unauthorized", { status: 401 });
	}

	const { token } = await res.json();

	// 쿠키 또는 localStorage 저장
	return new Response(null, {
		status: 302,
		headers: {
			"Set-Cookie": `token=${token}; Path=/; HttpOnly; Secure; SameSite=Lax`,
			Location: "/dashboard",
		},
	});
}
