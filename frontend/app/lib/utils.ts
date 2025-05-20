import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
	return twMerge(clsx(inputs));
}

export function getCookie(name = "token") {
	const value = `; ${document.cookie}`;
	console.log(value);
	const parts = value.split(`; ${name}=`);
	if (parts.length === 2) return parts.pop()?.split(";").shift();
}

export function fetcher(
	url: string,
	method = "GET",
	body: any = undefined,
) {
	const fetchOpt: RequestInit = {
		method: method,
		headers: {
			"Content-Type": "application/json",
		},
		credentials: "include", // 쿠키를 포함하여 요청
	};

	if (body) {
		fetchOpt.body = JSON.stringify(body);
	}

	return fetch(url, fetchOpt)
		.then((res) => {
			if (!res.ok) throw new Error(res.statusText);
			return res.json();
		})
		.catch((e: Error) => {
			console.log(e.message);
		});
}
