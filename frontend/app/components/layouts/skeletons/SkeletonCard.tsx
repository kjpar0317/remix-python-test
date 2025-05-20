import { Skeleton } from "~/components/ui/skeleton";

export default function SkeletonCard() {
	return (
		<div className="flex flex-col space-y-3 w-full items-center">
			<Skeleton className="h-[125px] w-11/12 rounded-xl" />
			<div className="space-y-2">
				<Skeleton className="h-4 w-11/12" />
				<Skeleton className="h-4 w-11/12" />
			</div>
		</div>
	);
}
