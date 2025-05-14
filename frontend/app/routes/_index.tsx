import type { MetaFunction } from "@remix-run/node";
import type { ActionFunction } from "@remix-run/node";

import { useRef } from "react";
import { useActionData, Form } from "@remix-run/react";
import html2canvas from "html2canvas-oklch";
import { jsPDF } from "jspdf";

import StockChart from "~/components/charts/StockChart";
import AnaysisContent from "~/components/features/AnalysisContent";
import { Input } from "~/components/ui/input";
import { Button } from "~/components/ui/button";

export const meta: MetaFunction = () => {
  return [
    { title: "New Remix App" },
    { name: "description", content: "Welcome to Remix!" },
  ];
};

export const action: ActionFunction = async ({ request }) => {
  const form = await request.formData();
  const ticker = form.get("ticker")?.toString();
  const timeframe = form.get("timeframe")?.toString();

  // Node fetch용 절대 URL 생성
  const urlObj = new URL(request.url);
  const apiUrl = new URL("/api/stock/chart-data", urlObj).href;

  const response = await fetch(apiUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ticker, timeframe }),
  });
  if (!response.ok)
    throw new Response("Failed to fetch chart data", {
      status: response.status,
    });
  const data = await response.json();

  return { ticker, timeframe, ...data };
};

export default function Index() {
  const actionData = useActionData<typeof action>();
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
          pageHeight
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
      <div className="flex flex-col items-stretch gap-16 w-full max-w-5xl mx-auto">
        <header className="flex flex-col items-center gap-9">
          <h1 className="leading text-2xl font-bold text-gray-800 dark:text-gray-100">
            Test
          </h1>
        </header>
        <Form method="post" className="mb-4 flex gap-2">
          <Input
            name="ticker"
            type="text"
            defaultValue="TSLY"
            placeholder="Ticker"
          />
          <Input
            name="timeframe"
            type="text"
            defaultValue="1mo"
            placeholder="Timeframe"
          />
          <Button type="submit">Load</Button>
        </Form>
        <Button onClick={handlePrint}>PDF</Button>
        <div ref={componentRef}>
          <div className="flex flex-col items-stretch">
            {actionData && <StockChart data={actionData} />}
          </div>
          {actionData && (
            <div className="flex flex-col items-stretch">
              <AnaysisContent
                ticker={actionData.ticker}
                timeframe={actionData.timeframe}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
