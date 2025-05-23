import { range } from "es-toolkit";
import OneLineSkeleton from "~/components/skeletons/OneLineSkeleton";
import TitleBlankCardSkeleton from "~/components/skeletons/TitleBlankCardSkeleton";

export default function StockChartSkeleton() {
    return (
        <>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                {range(4).map((num: number) => (
                    <OneLineSkeleton key={num.toString()} line={1} />
                ))}
            </div>
            <div className="w-full h-[300px]">
                <TitleBlankCardSkeleton className="h-[300px]"/>
            </div>
        </>
    );
}