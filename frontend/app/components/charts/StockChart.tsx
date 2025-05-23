import type {
	_DeepPartialArray,
	_DeepPartialObject,
} from "node_modules/chart.js/dist/types/utils";

import { useEffect, useRef } from "react";
import {
	Chart,
	Legend,
	LineController,
	LineElement,
	LinearScale,
	PointElement,
	ScatterController,
	TimeScale,
	Tooltip,
} from "chart.js";
import "chartjs-adapter-date-fns";
import annotationPlugin, {
	type AnnotationOptions,
	type AnnotationTypeRegistry,
} from "chartjs-plugin-annotation";
import { ko } from "date-fns/locale";

import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import { cn } from "~/lib/utils";

Chart.register(
	LineController,
	LineElement,
	PointElement,
	LinearScale,
	TimeScale,
	Tooltip,
	Legend,
	ScatterController,
	annotationPlugin,
);

type TunningPoints = {
	date: string;
	type: string;
	close: number;
};

interface StockChartProps {
	data: {
		dates: string[];
		close: number[];
		lstmClose: number[];
		cnnClose: number[];
		ma200: number[];
		rsi: number[];
		upperBand: number[];
		lowerBand: number[];
		macd: number[];
		signal: number[];
		sniperSignal: number[];
		tunningPoints: TunningPoints[];
		recommendMacdSignal: string;
		recommendGC: string;
		recommendRSI: string;
		recommendUpperLower: string;
		recommendTotalDecision: string;
	};
}

type AnnotationList =
	| _DeepPartialArray<AnnotationOptions<keyof AnnotationTypeRegistry>>
	| _DeepPartialObject<
			Record<string, AnnotationOptions<keyof AnnotationTypeRegistry>>
	  >
	| {
			type: string;
			scaleID: string;
			value: string;
			borderColor: string;
			borderWidth: number;
			label: { content: string; enabled: boolean; position: string };
	  }[]
	| undefined;

export default function StockChart({ data }: StockChartProps) {
	const chartRef = useRef<HTMLCanvasElement>(null);
	const chartInstance = useRef<Chart | null>(null);

	useEffect(() => {
		if (!chartRef.current) return;

		// 이전 차트 인스턴스 제거
		if (chartInstance.current) {
			chartInstance.current.destroy();
		}

		const ctx = chartRef.current.getContext("2d");
		if (!ctx) return;

		// const annotaionList: AnnotationList = [];

		// biome-ignore lint/complexity/noForEach: <explanation>
		// data.tunningPoints.forEach((pt: TunningPoints) => {
		// 	annotaionList.push({
		// 		type: "line",
		// 		scaleID: "x",
		// 		value: pt.date,
		// 		borderColor: pt.type === "buy" ? "green" : "red",
		// 		borderWidth: 1,
		// 		borderDash: [5, 5],
		// 		label: {
		// 			content: pt.type === "buy" ? "매수" : "매도",
		// 			enabled: true,
		// 			position: "start",
		// 		},
		// 	});
		// });

		const annotationList = [];

		for (let i = 0; i < data.tunningPoints.length - 1; i++) {
			const pt = data.tunningPoints[i];
			const nextPt = data.tunningPoints[i + 1];

			if (pt.type === "buy" && nextPt.type === "sell") {
				annotationList.push({
					type: "box",
					xMin: pt.date,
					xMax: nextPt.date,
					backgroundColor: "rgba(0, 255, 0, 0.05)", // 연한 초록색
					borderWidth: 0,
					// label: {
					// 	drawTime: 'afterDraw',
					// 	display: true,
					// 	content: "매수 구간",
					// 	position: {
					// 		x: 'center',
					// 		y: 'start'
					// 	}
					// },
				});
			} else if (pt.type === "sell" && nextPt.type === "buy") {
				annotationList.push({
					type: "box",
					xMin: pt.date,
					xMax: nextPt.date,
					backgroundColor: "rgba(255, 0, 0, 0.05)", // 연한 빨간색
					borderWidth: 0,
					// label: {
					// 	drawTime: 'afterDraw',
					// 	display: true,
					// 	content: "매도 구간",
					// 	position: {
					// 		x: 'center',
					// 		y: 'start'
					// 	}
					// },
				});
			}
		}

		console.log(annotationList);

		console.log(annotationList.reduce((acc: any, annotation, index) => {
			acc[`line${index}`] = annotation;
			return acc;
		}, {}));

		chartInstance.current = new Chart(ctx, {
			type: "line",
			data: {
				labels: data.dates,
				datasets: [
					{
						label: "주가",
						data: data.close,
						borderColor: "#3b82f6", // 밝은 파란색
						borderWidth: 2,
						fill: false,
						yAxisID: "y",
					},
					{
						label: "RSI",
						data: data.rsi,
						borderColor: "#ef4444", // 선명한 빨간색
						borderWidth: 1,
						fill: false,
						yAxisID: "y1",
					},
					{
						label: "상단밴드",
						data: data.upperBand,
						borderColor: "#10b981", // 에메랄드 그린
						borderWidth: 1,
						borderDash: [5, 5],
						fill: false,
						yAxisID: "y",
					},
					{
						label: "하단밴드",
						data: data.lowerBand,
						borderColor: "#14b8a6", // 틸 컬러
						borderWidth: 1,
						borderDash: [5, 5],
						fill: false,
						yAxisID: "y",
					},
					{
						label: "LSTM",
						data: data.lstmClose,
						borderColor: "#8b5cf6", // 보라색
						borderWidth: 1,
						borderDash: [5, 5],
						fill: false,
						yAxisID: "y",
					},
					{
						label: "CNN",
						data: data.cnnClose,
						borderColor: "#f59e0b", // 주황색
						borderWidth: 1,
						borderDash: [5, 5],
						fill: false,
						yAxisID: "y",
					},
					// {
					//   label: '스나이퍼 시그널',
					//   data: data.dates.map((date, i) => ({ x: date, y: data.sniperSignal[i] })),
					//   type: 'scatter',
					//   pointBackgroundColor: '#ffc658',
					//   showLine: false,
					//   yAxisID: 'ySignal'
					// }
				],
			},
			options: {
				responsive: true,
				interaction: {
					mode: "index",
					intersect: false,
				},
				scales: {
					x: {
						type: "time",
						time: {
							unit: "day",
							displayFormats: {
								day: "MM/dd",
							},
						},
						adapters: {
							date: {
								locale: ko,
							},
						},
						title: {
							display: true,
							text: "날짜",
						},
					},
					y: {
						type: "linear",
						position: "left",
						title: {
							display: true,
							text: "주가",
						},
					},
					y1: {
						type: "linear",
						position: "right",
						min: 0,
						max: 100,
						grid: {
							drawOnChartArea: false,
						},
						title: {
							display: true,
							text: "RSI",
						},
					},
					ySignal: {
						type: "linear",
						position: "right",
						display: false,
						min: 0,
						max: 1,
						grid: { drawOnChartArea: false },
					},
				},
				plugins: {
					tooltip: {
						mode: "index",
						intersect: false,
					},
					legend: {
						position: "top",
					},
					annotation: {
						common: {
							drawTime: "beforeDraw",
						},
						annotations: annotationList.reduce((acc: any, annotation, index) => {
							acc[`line${index}`] = annotation;
							return acc;
						}, {}),
					},
				},
			},
		});

		return () => {
			if (chartInstance.current) {
				chartInstance.current.destroy();
			}
		};
	}, [data]);

	return (
		<div className="w-full">
			<div className="grid grid-cols-4 gap-4 w-full">
				{[
					{ title: "MACD시그널", signal: data.recommendMacdSignal },
					{ title: "RSI", signal: data.recommendRSI },
					{ title: "볼린저밴드", signal: data.recommendUpperLower },
					{ title: "종합", signal: data.recommendTotalDecision },
				].map(({ title, signal }) => (
					<SignalCard key={title} title={title} signal={signal} />
				))}
			</div>
			<div className="bg-white p-4 rounded-lg shadow-lg">
				<canvas ref={chartRef} />
			</div>
		</div>
	);
}

function SignalCard({ title, signal }: { title: string; signal: string }) {
	const getSignalColor = (signal: string) => {
		switch (signal) {
			case "매수":
			case "강력매수":
				return "bg-red-100 text-red-800";
			case "매도":
			case "강력매도":
				return "bg-blue-100 text-blue-800";
			default:
				return "bg-gray-100 text-gray-800";
		}
	};

	return (
		<Card className="transition duration-500 transform hover:scale-105 hover:cursor-pointer">
			<CardHeader>
				<CardTitle>{title}</CardTitle>
			</CardHeader>
			<CardContent>
				<p className={cn(getSignalColor(signal))}>{signal}</p>
			</CardContent>
		</Card>
	);
}