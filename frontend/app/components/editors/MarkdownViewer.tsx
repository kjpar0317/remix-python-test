import ReactMarkdown from "react-markdown";

export default function MarkdownViewer({ content }: { content: string }) {
	return (
		<div className="text-base leading-relaxed space-y-4 px-4 py-6">
			<ReactMarkdown
				components={{
					h1: ({ node, ...props }) => (
						<h1 className="text-3xl font-bold mt-6 mb-4" {...props} />
					),
					h2: ({ node, ...props }) => (
						<h2 className="text-2xl font-semibold mt-6 mb-3" {...props} />
					),
					h3: ({ node, ...props }) => (
						<h3 className="text-xl font-medium mt-4 mb-2" {...props} />
					),
					p: ({ node, ...props }) => <p className="mb-2" {...props} />,
					ul: ({ node, ...props }) => (
						<ul className="list-disc pl-6 mb-2" {...props} />
					),
					li: ({ node, ...props }) => <li className="mb-1" {...props} />,
					strong: ({ node, ...props }) => (
						<strong className="font-semibold" {...props} />
					),
				}}
			>
				{content}
			</ReactMarkdown>
		</div>
	);
}
