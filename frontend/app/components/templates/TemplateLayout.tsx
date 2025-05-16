import Header from "~/components/layouts/common/Header";

export default function TemplateLayout({ children }: {
    children: React.ReactNode;
}) {
    return (
      <div>
        <Header />
        <main className="p-4">{children}</main>
      </div>
    );
}