import { range } from "es-toolkit";

import { cn } from "~/lib/utils";

export default function OneLineSkeleton({
	line,
	className,
}: { line?: number; className?: string }) {
	return (
		<>
			{range(line ?? 1).map((num: number) => (
				<div
					key={num.toString()}
					className={cn(
						"h-4 w-full animate-pulse rounded bg-gray-200 mb-2",
						className,
					)}
				/>
			))}
		</>
	);
}
