import OneLineSkeleton from "~/components/skeletons/OneLineSkeleton";
import TitleBlankCardSkeleton from "~/components/skeletons/TitleBlankCardSkeleton";

export default function AnalysisContentSkeleton() {
    return (
        <>
            <TitleBlankCardSkeleton className="w-full h-[150px]"/>
            <OneLineSkeleton className="w-full h-[200px]" />
            <TitleBlankCardSkeleton className="w-full h-[250px]" />
        </>
    );
}