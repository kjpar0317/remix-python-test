import type { ActionFunction } from "@remix-run/node";

export const action: ActionFunction = async () => {
  return new Response(null, {
    status: 302,
    headers: {
      "Set-Cookie": "token=; Path=/; HttpOnly; SameSite=Lax; Secure; Max-Age=0;",
      Location: "/login",
    },
  });
};