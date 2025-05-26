import type { ActionData } from "../schemas/dashboard";

import { Form, useActionData, useNavigation } from "@remix-run/react";
import html2canvas from "html2canvas-oklch";
import { jsPDF } from "jspdf";
import { useRef } from "react";

import StockChart from "~/components/charts/StockChart";
import AnaysisContent from "~/components/features/AnalysisContent";
import InnerLoading from "~/components/layouts/common/InnerLoading";
import { Button } from "~/components/ui/button";
import { Input } from "~/components/ui/input";
import {
	Select,
	SelectContent,
	SelectItem,
	SelectTrigger,
	SelectValue,
} from "~/components/ui/select";
import StockChartSkeleton from "~/components/charts/StockChartSkeleton";
import AnalysisContentSkeleton from "~/components/features/AnalysisContentSkeleton";

export default function Dashboard() {
	const actionData = useActionData<ActionData>();
	const navigation = useNavigation();
	const isLoading =
		navigation.state === "submitting" || navigation.state === "loading";
	const componentRef = useRef<HTMLDivElement>(null);

	async function handlePrint() {
		if (!componentRef.current) return;
		// 1) 화면 캡처
		const canvas = await html2canvas(componentRef.current);
		const img = canvas.toDataURL("image/png");

		// 2) jsPDF에 담기 (A4 비율 맞춤)
		const pdf = new jsPDF({ unit: "mm", format: "a4" });

		const pageWidth = pdf.internal.pageSize.getWidth(); // A4 페이지의 가로 크기
		const pageHeight = pdf.internal.pageSize.getHeight(); // A4 페이지의 세로 크기
		const imgWidth = canvas.width; // 캡처한 이미지의 너비
		const imgHeight = canvas.height; // 캡처한 이미지의 높이

		const imgRatio = pageWidth / imgWidth;
		const scaledImgHeight = imgHeight * imgRatio; // A4 페이지에 맞는 이미지의 높이

		let yOffset = 0; // y 위치 초기화

		// 이미지가 한 페이지에 맞는 경우
		if (scaledImgHeight <= pageHeight) {
			pdf.addImage(img, "PNG", 0, yOffset, pageWidth, scaledImgHeight);
		} else {
			// 이미지가 한 페이지를 넘어가는 경우
			while (yOffset < imgHeight) {
				// 한 페이지에 맞게 이미지 일부분을 그립니다.
				pdf.addImage(
					img,
					"PNG",
					0,
					yOffset * imgRatio, // yOffset을 이미지 비율에 맞게 조정
					pageWidth,
					pageHeight,
				);
				yOffset += pageHeight / imgRatio; // 다음 페이지로 넘어갈 위치 계산
				if (yOffset < imgHeight) pdf.addPage(); // 다음 페이지가 있다면 페이지 추가
			}
		}

		// PDF 저장
		pdf.save(`${actionData.ticker}_report.pdf`);
	}

	return (
		<div className="flex h-screen justify-center">
			<div className="flex flex-col items-stretch gap-1 w-full max-w-5xl mx-auto mt-3">
				<Form method="post" className="mb-1 flex gap-2">
				<Select name="currency" defaultValue="USD"> 
					<SelectTrigger> 
						<SelectValue /> 
					</SelectTrigger> 
					<SelectContent> 
						<SelectItem value="USD">USD</SelectItem> 
						<SelectItem value="KRW">KRW</SelectItem> 
						<SelectItem value="JPY">JPY</SelectItem> 
						<SelectItem value="EUR">EUR</SelectItem> 
						<SelectItem value="GBP">GBP</SelectItem> 
						<SelectItem value="CNY">CNY</SelectItem> 
					</SelectContent> 
				</Select>
					<Input
						name="ticker"
						type="text"
						defaultValue="TSLY"
						placeholder="Ticker"
					/>
					<Input
						name="timeframeValue"
						type="number"
						defaultValue="6"
						min="1"
						placeholder="Value"
						className="max-w-[4rem]"
					/>
					<Select name="timeframeUnit" defaultValue="mo">
						<SelectTrigger>
							<SelectValue />
						</SelectTrigger>
						<SelectContent>
							<SelectItem value="d">일</SelectItem>
							<SelectItem value="mo">달</SelectItem>
							<SelectItem value="y">년</SelectItem>
						</SelectContent>
					</Select>
					<Button type="submit">
						<InnerLoading isLoading={isLoading} />
						Load
					</Button>
				</Form>
				<Button onClick={handlePrint}>PDF</Button>
				<div ref={componentRef} className="w-full mt-5">
					<div className="flex flex-col items-stretch">
						{!isLoading ? (actionData && <StockChart data={actionData} />) : <StockChartSkeleton />}
					</div>
					{!isLoading ? (actionData && (
						<div className="flex flex-col items-stretch">
							<AnaysisContent
								ticker={actionData.ticker}
								timeframe={actionData.timeframe}
							/>
						</div>
					)) : <AnalysisContentSkeleton />
					}
				</div>
			</div>
		</div>
	);
}

export { action } from "~/services/server/dashboard.server";
