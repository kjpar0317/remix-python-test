import { useEffect, useRef } from 'react';
import {
  Chart,
  LineController,
  LineElement,
  PointElement,
  LinearScale,
  TimeScale,
  Tooltip,
  Legend,
  ScatterController
} from 'chart.js';
import 'chartjs-adapter-date-fns';
import { ko } from 'date-fns/locale';

import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "~/components/ui/card"
import { cn } from "~/lib/utils"

Chart.register(
  LineController,
  LineElement,
  PointElement,
  LinearScale,
  TimeScale,
  Tooltip,
  Legend,
  ScatterController
);

interface StockChartProps {
  data: {
    dates: string[];
    close: number[];
    rsi: number[];
    upperBand: number[];
    lowerBand: number[];
    sniperSignal: number[];
    recommendGC: string;
    recommendRSI: string;
    recommendUpperLower: string;
    recommendSniperSignal: string;
  };
}

export default function StockChart({ data }: StockChartProps) {
  const chartRef = useRef<HTMLCanvasElement>(null);
  const chartInstance = useRef<Chart | null>(null);

  useEffect(() => {
    if (!chartRef.current) return;

    // 이전 차트 인스턴스 제거
    if (chartInstance.current) {
      chartInstance.current.destroy();
    }

    const ctx = chartRef.current.getContext('2d');
    if (!ctx) return;

    chartInstance.current = new Chart(ctx, {
      type: 'line',
      data: {
        labels: data.dates,
        datasets: [
          {
            label: '주가',
            data: data.close,
            borderColor: '#2563eb',
            borderWidth: 2,
            fill: false,
            yAxisID: 'y'
          },
          {
            label: 'RSI',
            data: data.rsi,
            borderColor: '#dc2626',
            borderWidth: 1,
            fill: false,
            yAxisID: 'y1'
          },
          {
            label: '상단밴드',
            data: data.upperBand,
            borderColor: '#4ade80',
            borderWidth: 1,
            borderDash: [5, 5],
            fill: false,
            yAxisID: 'y'
          },
          {
            label: '하단밴드',
            data: data.lowerBand,
            borderColor: '#4ade80',
            borderWidth: 1,
            borderDash: [5, 5],
            fill: false,
            yAxisID: 'y'
          },
          {
            label: '스나이퍼 시그널',            
            data: data.dates.map((date, i) => ({ x: date, y: data.sniperSignal[i] })),
            type: 'scatter',
            pointBackgroundColor: '#ffc658',
            showLine: false,
            yAxisID: 'ySignal'
          }
        ]
      // biome-ignore lint/suspicious/noExplicitAny: <explanation>
      } as any,
      options: {
        responsive: true,
        interaction: {
          mode: 'index',
          intersect: false,
        },
        scales: {
          x: {
            type: 'time',
            time: {
              unit: 'day',
              displayFormats: {
                day: 'MM/dd'
              }
            },
            adapters: {
              date: {
                locale: ko
              }
            },
            title: {
              display: true,
              text: '날짜'
            }
          },
          y: {
            type: 'linear',
            position: 'left',
            title: {
              display: true,
              text: '주가'
            }
          },
          y1: {
            type: 'linear',
            position: 'right',
            min: 0,
            max: 100,
            grid: {
              drawOnChartArea: false
            },
            title: {
              display: true,
              text: 'RSI'
            }
          },
          ySignal: {
            type: 'linear',
            position: 'right',
            display: false,
            min: 0,
            max: 1,
            grid: { drawOnChartArea: false }
          }
        },
        plugins: {
          tooltip: {
            mode: 'index',
            intersect: false
          },
          legend: {
            position: 'top'
          }
        }
      }
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
          { title: "골든크로스", signal: data.recommendGC },
          { title: "RSI", signal: data.recommendRSI },
          { title: "볼린저밴드", signal: data.recommendUpperLower },
          { title: "스나이퍼", signal: data.recommendSniperSignal },
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
      case '매수':
      case '강력매수':
        return 'bg-red-100 text-red-800';
      case '매도':
      case '강력매도':
        return 'bg-blue-100 text-blue-800';
      default:
        return 'bg-gray-100 text-gray-800';
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