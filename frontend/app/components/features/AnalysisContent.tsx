import Chart from "chart.js/auto";
import { useEffect, useRef } from "react";

import MarkdownViewer from "~/components/editors/MarkdownViewer";
import SkeletonCard from "~/components/layouts/skeletons/SkeletonCard";
import useDashboard from "~/services/client/useDashboard";

interface StockAnalysisProps {
	ticker: string;
	timeframe: string;
}

export default function StockAnalysis({
	ticker,
	timeframe,
}: StockAnalysisProps) {
	const { analysisData, isTickerDetailLoading, setTickerDetail } =
		useDashboard();
	const chartRef = useRef<HTMLCanvasElement>(null);

	useEffect(() => {
		if (!ticker || !timeframe) return;
		setTickerDetail({ ticker, timeframe });
	}, [ticker, timeframe, setTickerDetail]);

	useEffect(() => {
		if (!analysisData) return;

		const ctx = chartRef.current?.getContext("2d");

		if (!ctx) return;

		const sentimentChart = new Chart(ctx, {
			type: "bar",
			data: {
				labels: ["단기 신뢰도", "장기 신뢰도", "뉴스 감성"],
				datasets: [
					{
						label: "투자 지표",
						data: [
							(analysisData?.recommendations &&
								analysisData?.recommendations.length > 0 &&
								analysisData?.recommendations[0].confidence * 100) ||
								0,
							(analysisData?.recommendations &&
								analysisData?.recommendations.length > 1 &&
								analysisData?.recommendations[1].confidence * 100) ||
								0,
							(analysisData?.news_sentiment?.overall_score &&
								analysisData?.news_sentiment?.overall_score * 100) ||
								0,
						],
						backgroundColor: [
							"rgba(255, 99, 132, 0.8)",
							"rgba(54, 162, 235, 0.8)",
							"rgba(255, 206, 86, 0.8)",
						],
						borderColor: [
							"rgba(255, 99, 132, 1)",
							"rgba(54, 162, 235, 1)",
							"rgba(255, 206, 86, 1)",
						],
						borderWidth: 1,
					},
				],
			},
			options: {
				responsive: true,
				plugins: {
					legend: {
						position: "top",
					},
					title: {
						display: true,
						text: "투자 신뢰도 분석",
					},
				},
				scales: {
					y: {
						beginAtZero: true,
						max: 100,
						ticks: {
							callback: (value) => `${value}%`,
						},
					},
				},
			},
		});

		return () => {
			sentimentChart.destroy();
		};
	}, [analysisData]);

	return (
		<div className="container mx-auto p-6">
			{(!isTickerDetailLoading && (
				<>
					<div className="bg-white rounded-lg shadow-lg p-6 mb-6">
						<h1 className="text-3xl font-bold text-gray-800 mb-2">
							{analysisData?.ticker} 주식 분석
						</h1>
						<div className="grid grid-cols-2 gap-4 mt-4">
							{analysisData?.recommendations?.map((rec) => (
								<div
									key={rec.type}
									className={`p-4 rounded-lg ${
										rec.action === "매수" ? "bg-red-50" : "bg-blue-50"
									}`}
								>
									<h3 className="text-lg font-semibold">
										{rec.type === "short_term" ? "단기 전망" : "장기 전망"}
									</h3>
									<div
										className={`text-2xl font-bold ${
											rec.action === "매수" ? "text-red-600" : "text-blue-600"
										}`}
									>
										{rec.action}
									</div>
									<div className="text-sm text-gray-600">
										신뢰도: {(rec.confidence * 100).toFixed(0)}%
									</div>
								</div>
							))}
						</div>
					</div>

					<div className="bg-white rounded-lg shadow-lg p-6 mb-6">
						<canvas ref={chartRef} height="100" />
					</div>

					<div className="bg-white rounded-lg shadow-lg p-6 mb-6">
						<h2 className="text-2xl font-bold mb-4">뉴스 감성 분석</h2>
						<div className="grid grid-cols-3 gap-4 mb-4">
							<div className="text-center p-4 bg-gray-50 rounded-lg">
								<div className="text-lg font-semibold">전체 점수</div>
								<div className="text-3xl font-bold text-blue-600">
									{(analysisData?.news_sentiment?.overall_score &&
										(analysisData?.news_sentiment?.overall_score * 100).toFixed(
											0,
										)) ||
										0}
									%
								</div>
							</div>
							<div className="text-center p-4 bg-gray-50 rounded-lg">
								<div className="text-lg font-semibold">감성</div>
								<div className="text-3xl font-bold text-gray-700">
									{analysisData?.news_sentiment?.sentiment_text}
								</div>
							</div>
							<div className="text-center p-4 bg-gray-50 rounded-lg">
								<div className="text-lg font-semibold">최근 트렌드</div>
								<div className="text-3xl font-bold text-gray-700">
									{analysisData?.news_sentiment?.recent_trend}
								</div>
							</div>
						</div>
						<div className="mt-4">
							<h3 className="text-lg font-semibold mb-2">주요 요인</h3>
							<div className="flex flex-wrap gap-2">
								{analysisData?.news_sentiment?.key_factors.map(
									(factor, index) => (
										<span
											key={index.toString()}
											className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full"
										>
											{factor}
										</span>
									),
								)}
							</div>
						</div>
					</div>

					{/* 상세 분석 섹션 */}
					<div className="bg-white rounded-lg shadow-lg p-6">
						<h2 className="text-2xl font-bold mb-4">상세 분석</h2>
						<div className="prose max-w-none">
							<MarkdownViewer content={analysisData?.analysis || ""} />
						</div>
					</div>
				</>
			)) || <SkeletonCard />}
		</div>
	);
}
