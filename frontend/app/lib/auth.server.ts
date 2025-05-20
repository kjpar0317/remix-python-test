export function getToken(request: Request) {
	const cookie = request.headers.get("cookie");

	if (!cookie) return null;

	const token = Object.fromEntries(
		cookie.split("; ").map((c) => c.split("=")),
	).token;

	if (!token) return null;

	return token;
}
