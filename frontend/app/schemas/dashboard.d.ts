export type ActionData = ReturnType<typeof action>;

export interface StockAnalysisData {
	ticker: string;
	analysis: string;
	recommendations: Array<{
		type: string;
		action: string;
		confidence: number;
	}>;
	news_sentiment: {
		overall_score: number;
		sentiment_text: string;
		recent_trend: string;
		key_factors: string[];
	};
}
