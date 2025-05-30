import type { LoaderFunctionArgs, MetaFunction } from "@remix-run/node";
import { redirect } from "@remix-run/node";
import { getToken } from "~/lib/auth.server";

export const meta: MetaFunction = () => {
	return [
		{ title: "New Remix App" },
		{ name: "description", content: "Welcome to Remix!" },
	];
};

export async function loader({ request }: LoaderFunctionArgs) {
	const token = getToken(request);

	console.log(`token: ${token}`);

	if (token) {
		return redirect("/dashboard");
	}
	// return redirect("/login");
}

export default function Index() {
	return <div className="flex h-screen justify-center">Index</div>;
}
