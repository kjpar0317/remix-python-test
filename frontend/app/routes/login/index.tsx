import type { ActionData } from "./+types";

import { Form, useActionData} from "@remix-run/react";

import { action } from "./action";
import { Label } from "~/components/ui/label";
import { Input } from "~/components/ui/input";
import { Button } from "~/components/ui/button";

export default function Login() {
    const actionData = useActionData<ActionData>();

    return (
        <div className="flex items-center justify-center w-full h-screen flex-col gap-4">
            <Form method="post" className="w-2/5 space-y-3">
                <Label htmlFor="email" >Email</Label>
                <Input name="email" defaultValue="test"/>
                <Label htmlFor="passwd">Password</Label>
                <Input name="passwd" type="password" defaultValue="test"/>
                <Button type="submit" className="w-full">Login</Button>
                {actionData?.error && <p>{actionData.error}</p>}
            </Form>
        </div>
    );
}

export const handle = { noLayout: true };
export { action };