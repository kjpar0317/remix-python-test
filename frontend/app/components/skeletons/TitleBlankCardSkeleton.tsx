import { range } from "es-toolkit";

import { cn } from "~/lib/utils";

export default function TitleBlankCardSkeleton({
	dupnum,
	className,
}: { dupnum?: number; className?: string }) {
	return (
		<>
			{range(dupnum ?? 1).map((num: number) => (
				<div
					key={num.toString()}
					className={cn(
						"w-full rounded-2xl shadow-lg animate-pulse space-y-2 border-1",
						className,
					)}
				>
					<div className="mt-3 ml-2 mr-2 rounded-2xl py-4 h-8 bg-gray-200">
						&nbsp;
					</div>
					<div className="flex h-[calc(100%_-_60px)] justify-between px-5 mb-2 ml-2 mr-2 rounded-2xl bg-gray-200">
						&nbsp;
					</div>
				</div>
			))}
		</>
	);
}
