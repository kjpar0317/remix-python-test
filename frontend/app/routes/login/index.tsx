import type { ActionData } from "./+types";

import { Form, useActionData, useNavigation } from "@remix-run/react";

import { action } from "./action";
import { cn } from "~/lib/utils";
import { Label } from "~/components/ui/label";
import { Input } from "~/components/ui/input";
import { Button } from "~/components/ui/button";
import InnerLoading from "~/components/layouts/common/InnerLoading";

export default function Login() {
    const actionData = useActionData<ActionData>();
    const navigation = useNavigation();
    const isLoading = navigation.state === "submitting" || navigation.state === "loading";

    return (
        <div className="h-screen bg-gradient-to-tl from-green-400 to-indigo-900 w-full py-22 px-4 perspective-distant overflow-hidden">
            <div className="flex flex-col items-center justify-center animate-rotate-3d">
                <svg
                    className="focus:outline-none"
                    aria-label="logo"
                    role="banner"
                    width="188"
                    height="74"
                    viewBox="0 0 188 74"
                    fill="none"
                    xmlns="http://www.w3.org/2000/svg"
                >
                    <path
                        d="M69 17.0551C69.0331 25.2688 75.7248 32 83.9453 32C92.1861 32 98.8902 25.2959 98.8902 17.0551V14.9453C98.8902 11.5521 101.651 8.79118 105.044 8.79118C108.438 8.79118 111.199 11.5521 111.199 14.9453C111.199 15.9163 111.986 16.7037 112.957 16.7037H118.232C119.203 16.7037 119.99 15.9163 119.99 14.9453C119.99 6.70457 113.286 0 105.045 0C96.8041 0 90.0995 6.70457 90.0995 14.9453V17.0551C90.0995 20.4489 87.3386 23.2088 83.9458 23.2088C80.5526 23.2088 77.7917 20.4489 77.7917 17.0551C77.7917 16.0842 77.0043 15.2968 76.0333 15.2968H70.7583C69.7874 15.2973 69 16.0842 69 17.0551Z"
                        fill="white"
                    />
                </svg>
                <h1 className="text-white text-sm sm:text-2xl lg:text-3xl">
                    Test
                </h1>

                <div className="bg-white shadow rounded lg:w-[600px] md:w-1/2 w-full p-10 mt-5 md:mt-16 lg:mt-21 z-2">
                    <p className="focus:outline-none text-2xl font-extrabold leading-6 text-gray-800">
                        LOGIN
                    </p>
                    <p className="focus:outline-none text-sm mt-4 mb-8 font-medium leading-none text-gray-500">
                        Login to AI Stock Test
                    </p>
                    <Form method="post">
                        <div>
                            <Label
                                htmlFor="email"
                                className="text-sm font-medium leading-none text-gray-800 mb-1"
                            >
                                ID
                            </Label>
                            <Input name="email" defaultValue="test" placeholder="이메일을 입력하여 주세요."/>
                        </div>
                        <div className="mt-6 w-full">
                            <Label 
                                htmlFor="passwd"
                                className="text-sm font-medium leading-none text-gray-800 mb-1"
                            >
                                Password
                            </Label>
                            <div className="relative flex items-center justify-center">
                                <Input name="passwd" type="password" defaultValue="test" placeholder="비밀번호를 입력하여 주세요."/>
                                <div className="absolute right-0 mt-2 mr-3 cursor-pointer">
                                    <svg
                                        role="banner"
                                        xmlns="http://www.w3.org/2000/svg"
                                        width="16"
                                        height="16"
                                        viewBox="0 0 16 16"
                                        fill="none"
                                    >
                                        <path
                                            d="M7.99978 2C11.5944 2 14.5851 4.58667 15.2124 8C14.5858 11.4133 11.5944 14 7.99978 14C4.40511 14 1.41444 11.4133 0.787109 8C1.41378 4.58667 4.40511 2 7.99978 2ZM7.99978 12.6667C9.35942 12.6664 10.6787 12.2045 11.7417 11.3568C12.8047 10.509 13.5484 9.32552 13.8511 8C13.5473 6.67554 12.8031 5.49334 11.7402 4.64668C10.6773 3.80003 9.35864 3.33902 7.99978 3.33902C6.64091 3.33902 5.32224 3.80003 4.25936 4.64668C3.19648 5.49334 2.45229 6.67554 2.14844 8C2.45117 9.32552 3.19489 10.509 4.25787 11.3568C5.32085 12.2045 6.64013 12.6664 7.99978 12.6667ZM7.99978 11C7.20413 11 6.44106 10.6839 5.87846 10.1213C5.31585 9.55871 4.99978 8.79565 4.99978 8C4.99978 7.20435 5.31585 6.44129 5.87846 5.87868C6.44106 5.31607 7.20413 5 7.99978 5C8.79543 5 9.55849 5.31607 10.1211 5.87868C10.6837 6.44129 10.9998 7.20435 10.9998 8C10.9998 8.79565 10.6837 9.55871 10.1211 10.1213C9.55849 10.6839 8.79543 11 7.99978 11ZM7.99978 9.66667C8.4418 9.66667 8.86573 9.49107 9.17829 9.17851C9.49085 8.86595 9.66644 8.44203 9.66644 8C9.66644 7.55797 9.49085 7.13405 9.17829 6.82149C8.86573 6.50893 8.4418 6.33333 7.99978 6.33333C7.55775 6.33333 7.13383 6.50893 6.82126 6.82149C6.5087 7.13405 6.33311 7.55797 6.33311 8C6.33311 8.44203 6.5087 8.86595 6.82126 9.17851C7.13383 9.49107 7.55775 9.66667 7.99978 9.66667Z"
                                            fill="#71717A"
                                        />
                                    </svg>
                                </div>
                            </div>
                        </div>
                        <div className="mt-8">
                            {actionData?.error && <p className="text-red-500">{actionData.error}</p>}
                            <Button
                                type="submit"
                                className={cn(
                                    "focus:ring-2 focus:ring-offset-2 focus:ring-indigo-700 text-sm font-semibold leading-none text-white focus:outline-none bg-gradient-to-tr from-indigo-700 to-indigo-400 border rounded hover:bg-indigo-600 w-full",
                                    isLoading ? "py-3" : "py-4",
                                )}
                            >
                                <InnerLoading isLoading={isLoading} />
                                LOGIN
                            </Button>
                        </div>
                    </Form>
                </div>
            </div>
            <ul className="circles relative w-full h-full overflow-hidden z-1">
                <li className="circle circle1" />
                <li className="circle circle2" />
                <li className="circle circle3" />
                <li className="circle circle4" />
                <li className="circle circle5" />
                <li className="circle circle6" />
                <li className="circle circle7" />
                <li className="circle circle8" />
                <li className="circle circle9" />
                <li className="circle circle10" />
            </ul>
        </div>        
    );
}

export const handle = { noLayout: true };
export { action };