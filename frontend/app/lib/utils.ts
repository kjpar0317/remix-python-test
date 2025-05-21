import type { ActionFunctionArgs } from "@remix-run/node";

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
	const apiUrl = process.env.NODE_ENV === "production" ? process.env.API_BASE_URL + url.replace(/^\/api/, ""): url;
	const requestInit: RequestInit = {
		method: method,
		headers: {
			"Content-Type": "application/json",
		},
		credentials: "include", // 쿠키를 포함하여 요청
	};

	if (body) {
		requestInit.body = JSON.stringify(body);
	}

	return fetch(apiUrl, requestInit)
		.then((res) => {
			if (!res.ok) throw new Error(res.statusText);
			return res.json();
		})
		.catch((e: Error) => {
			console.log(e.message);
		});
}

export function ssrFetcher(
	request: Request,
	url: string,
	method = "GET",
	body: any = undefined
) {
	const apiUrl = process.env.API_BASE_URL + url.replace(/^\/api/, "");
	let requestInit: RequestInit = {
		method: method,
	};

	// clientside는 cookie를 포함하여 요청하고 serverside는 Cookie에 직접 set해서 요청
	if (body instanceof FormData) {
		requestInit = {
			...requestInit,
			headers: {
				Cookie: request.headers.get("cookie") ?? "",
			},
			body: body,
		};
	} else if (body) {
		requestInit = {
			...requestInit,
			headers: {
				"Content-Type": "application/json",
				Cookie: request.headers.get("cookie") ?? "",
			},
			body: JSON.stringify(body),
		};
	}

	return fetch(apiUrl, requestInit)
		.then((res) => {
			if (!res.ok) throw new Error(res.statusText);
			return res.json();
		})
		.catch((e: Error) => {
			console.log(e.message);
		});
}
