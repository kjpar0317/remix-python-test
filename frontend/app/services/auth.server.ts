import { parse } from "cookie";

export async function getToken(request: Request) {
  const cookie = request.headers.get("Cookie");
  const cookies = parse(cookie || "");
  return cookies.token;
}