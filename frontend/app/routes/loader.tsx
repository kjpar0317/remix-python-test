
import type { LoaderFunctionArgs } from "@remix-run/node";
import { redirect } from "@remix-run/node";
import { getToken } from "~/services/auth/auth.server";  

export async function loader({ request }: LoaderFunctionArgs) {
  const token = getToken(request);
  if(token) {
    return redirect('/dashboard');
  }
  return redirect('/login');
}