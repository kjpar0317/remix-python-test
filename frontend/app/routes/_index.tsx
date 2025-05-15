import type { MetaFunction } from "@remix-run/node";

import { loader } from "./loader";

export const meta: MetaFunction = () => {
  return [
    { title: "New Remix App" },
    { name: "description", content: "Welcome to Remix!" },
  ];
};

export default function Index() {
  return (
    <div className="flex h-screen justify-center">
      Index
    </div>
  );
}

export const handle = { noLayout: true };
export { loader };