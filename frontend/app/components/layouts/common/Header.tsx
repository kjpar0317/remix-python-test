import { Form } from "@remix-run/react";

import { Button } from "~/components/ui/button";
import {
	Tooltip,
	TooltipContent,
	TooltipProvider,
	TooltipTrigger,
} from "~/components/ui/tooltip";

export default function Header() {
	return (
		<nav className="fixed right-3 top-0 z-[0] flex w-full flex-row items-center justify-between rounded-lg bg-white/30 py-2 backdrop-blur-xl transition-all dark:bg-transparent md:right-[30px] md:top-4 md:w-[calc(100vw_-_8%)] md:p-2 lg:w-[calc(100vw_-_6%)] xl:top-[20px] xl:w-[calc(100vw_-_365px)] 2xl:w-[calc(100vw_-_380px)]">
			<div className="ml-[20px] md:ml-[2px]">
				<h1 className="animate-text bg-gradient-to-r from-pink-500 via-purple-500 to-blue-500 shrink capitalize font-bold bg-clip-text text-transparent text-2xl font-black">
					TEST
				</h1>
			</div>
			<div className="w-[154px] min-w-max md:ml-auto">
				<div className="flex min-w-max max-w-max flex-grow justify-around gap-1 rounded-lg md:px-2 md:py-2 md:pl-3 xl:gap-2">
					<Button className="items-center justify-center whitespace-nowrap font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 border bg-background hover:bg-accent hover:text-accent-foreground flex h-9 min-w-9 cursor-pointer rounded-full border-zinc-200 p-0 text-xl text-zinc-950 dark:border-zinc-800 dark:text-white md:min-h-10 md:min-w-10">
						<svg
							stroke="currentColor"
							fill="none"
							strokeWidth="1.5"
							viewBox="0 0 24 24"
							aria-hidden="true"
							className="h-4 w-4 stroke-2"
							height="1em"
							width="1em"
							xmlns="http://www.w3.org/2000/svg"
						>
							<path
								strokeLinecap="round"
								strokeLinejoin="round"
								d="M21.752 15.002A9.72 9.72 0 0 1 18 15.75c-5.385 0-9.75-4.365-9.75-9.75 0-1.33.266-2.597.748-3.752A9.753 9.753 0 0 0 3 11.25C3 16.635 7.365 21 12.75 21a9.753 9.753 0 0 0 9.002-5.998Z"
							/>
						</svg>
					</Button>
					<TooltipProvider>
						<Tooltip>
							<TooltipTrigger asChild>
								<Form method="post" action="/logout">
									<button
										type="submit"
										className="items-center justify-center whitespace-nowrap font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 border bg-background hover:bg-accent hover:text-accent-foreground flex h-9 min-w-9 cursor-pointer rounded-full border-zinc-200 p-0 text-xl text-zinc-950 dark:border-zinc-800 dark:text-white md:min-h-10 md:min-w-10"
									>
										<svg
											stroke="currentColor"
											fill="none"
											strokeWidth="1.5"
											viewBox="0 0 24 24"
											aria-hidden="true"
											className="h-4 w-4 stroke-2 text-zinc-950 dark:text-white"
											height="1em"
											width="1em"
											xmlns="http://www.w3.org/2000/svg"
										>
											<path
												strokeLinecap="round"
												strokeLinejoin="round"
												d="M15.75 9V5.25A2.25 2.25 0 0 0 13.5 3h-6a2.25 2.25 0 0 0-2.25 2.25v13.5A2.25 2.25 0 0 0 7.5 21h6a2.25 2.25 0 0 0 2.25-2.25V15m3 0 3-3m0 0-3-3m3 3H9"
											/>
										</svg>
									</button>
								</Form>
							</TooltipTrigger>
							<TooltipContent>
								<p>LOGOUT</p>
							</TooltipContent>
						</Tooltip>
					</TooltipProvider>
				</div>
			</div>
		</nav>
	);
}
