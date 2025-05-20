import type { StockAnalysisData } from "~/schemas/dashboard";

import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

import { fetcher } from "~/lib/utils";

export default function useDashboard() {
	const [tickerDetail, setTickerDetail] = useState<{
		ticker: string;
		timeframe: string;
	}>();
	const { data: analysisData, isLoading: isTickerDetailLoading } =
		useQuery<StockAnalysisData>({
			queryKey: ["tickerDetail", tickerDetail],
			queryFn: () => fetcher("/api/stock/analysis", "POST", tickerDetail),
			enabled: !!tickerDetail,
		});

	return {
		analysisData,
		isTickerDetailLoading,
		setTickerDetail,
	};
}
