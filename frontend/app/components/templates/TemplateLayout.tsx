export default function TemplateLayout({ children }: {
    children: React.ReactNode;
}) {
    return (
      <div>
        <header className="p-4 bg-gray-800 text-white flex justify-between">
          <div>로고</div>
        </header>
        <main className="p-4">{children}</main>
      </div>
    );
}